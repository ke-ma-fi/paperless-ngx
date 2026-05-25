import json
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from llama_index.core.schema import NodeWithScore
from llama_index.core.schema import TextNode

from paperless_ai.chat import CHAT_METADATA_DELIMITER
from paperless_ai.chat import CHAT_RETRIEVER_TOP_K
from paperless_ai.chat import CHAT_SCOPED_OVERSAMPLE_FACTOR
from paperless_ai.chat import stream_chat_with_documents


@pytest.fixture(autouse=True)
def patch_embed_model():
    from llama_index.core import settings as llama_settings
    from llama_index.core.embeddings.mock_embed_model import MockEmbedding

    # Use a real BaseEmbedding subclass to satisfy llama-index 0.14 validation
    llama_settings.Settings.embed_model = MockEmbedding(embed_dim=1536)
    yield
    llama_settings.Settings.embed_model = None


@pytest.fixture
def mock_document():
    doc = MagicMock()
    doc.pk = 1
    doc.title = "Test Document"
    doc.filename = "test_file.pdf"
    doc.content = "This is the document content."
    return doc


def assert_chat_output(
    output: list[str],
    *,
    expected_chunks: list[str],
    expected_references: list[dict[str, int | str]],
) -> None:
    assert output[:-1] == expected_chunks

    trailer = output[-1]
    assert trailer.startswith(CHAT_METADATA_DELIMITER)
    assert json.loads(trailer.removeprefix(CHAT_METADATA_DELIMITER)) == {
        "references": expected_references,
    }


def _node_with_score(
    document_id: str,
    title: str,
    text: str = "content",
    score: float = 0.9,
) -> NodeWithScore:
    node = TextNode(
        text=text,
        metadata={"document_id": document_id, "title": title},
    )
    return NodeWithScore(node=node, score=score)


def test_stream_chat_with_one_document_retrieval(mock_document) -> None:
    """Not scoped: the only node in the index belongs to the selected document."""
    result = _node_with_score(str(mock_document.pk), "Test Document")

    with (
        patch("paperless_ai.chat.AIClient") as mock_client_cls,
        patch("paperless_ai.chat.load_or_build_index") as mock_load_index,
        patch("llama_index.core.retrievers.VectorIndexRetriever") as mock_retriever_cls,
        patch(
            "llama_index.core.query_engine.RetrieverQueryEngine.from_args",
        ) as mock_query_engine_cls,
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.llm = MagicMock()

        # Only one node in the index and it matches — not scoped
        mock_index = MagicMock()
        mock_index.docstore.docs = {result.node.node_id: result.node}
        mock_load_index.return_value = mock_index

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [result]
        mock_retriever_cls.return_value = mock_retriever

        mock_response_stream = MagicMock()
        mock_response_stream.response_gen = iter(["chunk1", "chunk2"])
        mock_query_engine = MagicMock()
        mock_query_engine_cls.return_value = mock_query_engine
        mock_query_engine.query.return_value = mock_response_stream

        output = list(stream_chat_with_documents("What is this?", [mock_document]))

        # Not scoped → retrieve_k == CHAT_RETRIEVER_TOP_K (no oversampling)
        mock_retriever_cls.assert_called_once_with(
            index=mock_index,
            similarity_top_k=CHAT_RETRIEVER_TOP_K,
        )
        mock_query_engine.query.assert_called_once_with("What is this?")
        assert_chat_output(
            output,
            expected_chunks=["chunk1", "chunk2"],
            expected_references=[
                {"id": mock_document.pk, "title": "Test Document"},
            ],
        )


def test_stream_chat_with_multiple_documents_retrieval() -> None:
    """Scoped: two docs selected, a third doc's node is in the index and must be filtered out."""
    doc1 = MagicMock(pk=1, title="Document 1", filename="doc1.pdf")
    doc2 = MagicMock(pk=2, title="Document 2", filename="doc2.pdf")

    result_doc1 = _node_with_score("1", "Document 1", score=0.95)
    # Second node for doc1 — tests that references are deduplicated
    result_doc1_dup = _node_with_score("1", "Document 1 Duplicate", score=0.85)
    result_doc2 = _node_with_score("2", "Document 2", score=0.80)
    # Foreign node returned by FAISS but must not appear in top_nodes or references
    result_foreign = _node_with_score("3", "Document 3", score=0.75)

    # Docstore has 4 nodes: doc1 (×2), doc2, doc3 — only doc1+doc2 selected → scoped
    all_docstore = {
        result_doc1.node.node_id: result_doc1.node,
        result_doc1_dup.node.node_id: result_doc1_dup.node,
        result_doc2.node.node_id: result_doc2.node,
        result_foreign.node.node_id: result_foreign.node,
    }

    with (
        patch("paperless_ai.chat.AIClient") as mock_client_cls,
        patch("paperless_ai.chat.load_or_build_index") as mock_load_index,
        patch("llama_index.core.retrievers.VectorIndexRetriever") as mock_retriever_cls,
        patch(
            "llama_index.core.query_engine.RetrieverQueryEngine.from_args",
        ) as mock_query_engine_cls,
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.llm = MagicMock()

        mock_index = MagicMock()
        mock_index.docstore.docs = all_docstore
        mock_load_index.return_value = mock_index

        mock_retriever = MagicMock()
        # FAISS returns all four; post-filter must drop the foreign one
        mock_retriever.retrieve.return_value = [
            result_doc1,
            result_doc1_dup,
            result_doc2,
            result_foreign,
        ]
        mock_retriever_cls.return_value = mock_retriever

        mock_response_stream = MagicMock()
        mock_response_stream.response_gen = iter(["chunk1", "chunk2"])
        mock_query_engine = MagicMock()
        mock_query_engine_cls.return_value = mock_query_engine
        mock_query_engine.query.return_value = mock_response_stream

        output = list(stream_chat_with_documents("What's up?", [doc1, doc2]))

        # Scoped → retrieve_k uses oversampling factor
        expected_k = min(
            len(all_docstore),
            CHAT_RETRIEVER_TOP_K * CHAT_SCOPED_OVERSAMPLE_FACTOR,
        )
        mock_retriever_cls.assert_called_once_with(
            index=mock_index,
            similarity_top_k=expected_k,
        )
        mock_query_engine.query.assert_called_once_with("What's up?")
        assert_chat_output(
            output,
            expected_chunks=["chunk1", "chunk2"],
            # doc1 appears once despite two matching nodes (deduplication)
            # doc3 absent despite being in FAISS results (post-filter)
            expected_references=[
                {"id": 1, "title": "Document 1"},
                {"id": 2, "title": "Document 2"},
            ],
        )


def test_stream_chat_no_matching_nodes() -> None:
    with (
        patch("paperless_ai.chat.AIClient") as mock_client_cls,
        patch("paperless_ai.chat.load_or_build_index") as mock_load_index,
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.llm = MagicMock()

        mock_index = MagicMock()
        # Empty index → matching_node_ids is empty → early return
        mock_index.docstore.docs = {}
        mock_load_index.return_value = mock_index

        output = list(stream_chat_with_documents("Any info?", [MagicMock(pk=1)]))

        assert output == ["Sorry, I couldn't find any content to answer your question."]


def test_stream_chat_scoped_empty_after_filter() -> None:
    """FAISS returns results but none belong to the selected document — 'no content' reply."""
    doc1 = MagicMock(pk=1, title="Document 1", filename="doc1.pdf")

    doc1_node = _node_with_score("1", "Document 1").node
    # This node is in the index but NOT selected; FAISS returns it, post-filter drops it
    foreign_result = _node_with_score("2", "Document 2")

    with (
        patch("paperless_ai.chat.AIClient") as mock_client_cls,
        patch("paperless_ai.chat.load_or_build_index") as mock_load_index,
        patch("llama_index.core.retrievers.VectorIndexRetriever") as mock_retriever_cls,
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_index = MagicMock()
        mock_index.docstore.docs = {
            doc1_node.node_id: doc1_node,
            foreign_result.node.node_id: foreign_result.node,
        }
        mock_load_index.return_value = mock_index

        mock_retriever = MagicMock()
        # FAISS only returns the foreign node — doc1 content is not in the candidates
        mock_retriever.retrieve.return_value = [foreign_result]
        mock_retriever_cls.return_value = mock_retriever

        output = list(stream_chat_with_documents("specific question", [doc1]))

        assert output == ["Sorry, I couldn't find any content to answer your question."]
