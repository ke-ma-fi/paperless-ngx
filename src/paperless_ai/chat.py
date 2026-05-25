import json
import logging
import sys

from documents.models import Document
from paperless_ai.client import AIClient
from paperless_ai.indexing import get_rag_prompt_helper
from paperless_ai.indexing import load_or_build_index

logger = logging.getLogger("paperless_ai.chat")

CHAT_METADATA_DELIMITER = "\n\n__PAPERLESS_CHAT_METADATA__"
MAX_CHAT_REFERENCES = 3
CHAT_RETRIEVER_TOP_K = 5
CHAT_SCOPED_OVERSAMPLE_FACTOR = 20

CHAT_PROMPT_TMPL = """Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {query_str}
    Answer:"""


class _FixedNodeRetriever:
    """Returns pre-fetched, pre-filtered nodes to the query engine without re-querying."""

    def __init__(self, nodes: list) -> None:
        self._nodes = nodes

    def retrieve(self, *args, **kwargs) -> list:
        return self._nodes


def _build_document_reference(
    document: Document,
    title: str | None = None,
) -> dict[str, int | str]:
    return {
        "id": document.pk,
        "title": title or document.title or document.filename,
    }


def _get_document_references(
    documents: list[Document],
    top_nodes: list,
) -> list[dict[str, int | str]]:
    allowed_documents = {doc.pk: doc for doc in documents}
    references: list[dict[str, int | str]] = []
    seen_document_ids: set[int] = set()

    for node in top_nodes:
        try:
            document_id = int(node.metadata["document_id"])
        except (KeyError, TypeError, ValueError):  # pragma: no cover
            continue

        if document_id in seen_document_ids or document_id not in allowed_documents:
            continue

        seen_document_ids.add(document_id)
        document = allowed_documents[document_id]
        references.append(
            _build_document_reference(document, node.metadata.get("title")),
        )

        if len(references) >= MAX_CHAT_REFERENCES:  # pragma: no cover
            break

    return references


def _format_chat_metadata_trailer(references: list[dict[str, int | str]]) -> str:
    return (
        f"{CHAT_METADATA_DELIMITER}"
        f"{json.dumps({'references': references}, separators=(',', ':'))}"
    )


def stream_chat_with_documents(query_str: str, documents: list[Document]):
    from llama_index.core.prompts import PromptTemplate
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.response_synthesizers import get_response_synthesizer
    from llama_index.core.retrievers import VectorIndexRetriever

    client = AIClient()
    index = load_or_build_index()

    doc_ids = {str(doc.pk) for doc in documents}
    total_nodes = len(index.docstore.docs)

    # Collect node IDs for the selected documents.
    # FAISS has no pre-filter support; we build this set for post-filtering after retrieval.
    matching_node_ids = {
        node.node_id
        for node in index.docstore.docs.values()
        if node.metadata.get("document_id") in doc_ids
    }

    if not matching_node_ids:
        logger.warning("No nodes found for the given documents.")
        yield "Sorry, I couldn't find any content to answer your question."
        return

    scoped = len(matching_node_ids) < total_nodes
    # Over-fetch when scoping so the post-filter has enough candidates.
    # IndexFlatL2 always scans every vector regardless of k, so a larger k
    # adds only minor overhead while dramatically improving recall within the subset.
    retrieve_k = (
        min(total_nodes, CHAT_RETRIEVER_TOP_K * CHAT_SCOPED_OVERSAMPLE_FACTOR)
        if scoped
        else CHAT_RETRIEVER_TOP_K
    )

    retriever = VectorIndexRetriever(index=index, similarity_top_k=retrieve_k)
    all_results = retriever.retrieve(query_str)

    top_nodes = (
        [n for n in all_results if n.node.node_id in matching_node_ids][:CHAT_RETRIEVER_TOP_K]
        if scoped
        else all_results
    )

    if not top_nodes:
        if scoped:
            # The selected documents have no nodes among the global top candidates.
            # This happens when those documents are semantically distant from the query.
            logger.warning(
                "Retriever returned no nodes within the %d selected documents "
                "(oversample_k=%d). The document content may be semantically "
                "distant from the query.",
                len(doc_ids),
                retrieve_k,
            )
        else:
            logger.warning("Retriever returned no nodes for the given documents.")
        yield "Sorry, I couldn't find any content to answer your question."
        return

    references = _get_document_references(documents, top_nodes)

    prompt_template = PromptTemplate(template=CHAT_PROMPT_TMPL)
    response_synthesizer = get_response_synthesizer(
        llm=client.llm,
        prompt_helper=get_rag_prompt_helper(),
        text_qa_template=prompt_template,
        streaming=True,
    )

    # Pass pre-filtered nodes directly so the query engine uses only scoped context
    # and does not issue a second retrieval or embedding call.
    query_engine = RetrieverQueryEngine.from_args(
        retriever=_FixedNodeRetriever(top_nodes),
        llm=client.llm,
        response_synthesizer=response_synthesizer,
        streaming=True,
    )

    logger.debug("Document chat query: %s", query_str)

    response_stream = query_engine.query(query_str)

    for chunk in response_stream.response_gen:
        yield chunk
        sys.stdout.flush()

    if references:
        yield _format_chat_metadata_trailer(references)
