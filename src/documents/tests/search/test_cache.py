from __future__ import annotations

import pytest
from django.contrib.auth.models import User

from documents.caching import bump_search_cache_generation
from documents.caching import get_search_results_cache
from documents.caching import set_search_results_cache
from documents.models import Document
from documents.search._backend import SearchMode
from documents.search._backend import SearchResults
from documents.search._backend import TantivyBackend

pytestmark = [pytest.mark.search, pytest.mark.django_db]


def _make_results(query: str = "test") -> SearchResults:
    return SearchResults(hits=[], total=0, query=query)


class TestSearchCacheFunctions:
    """Unit tests for the search cache helpers in caching.py."""

    def test_miss_returns_none(self) -> None:
        result = get_search_results_cache(
            "missing",
            "text",
            1,
            None,
            sort_reverse=False,
        )
        assert result is None

    def test_set_then_get_returns_results(self) -> None:
        results = _make_results("invoice")
        set_search_results_cache(
            "invoice",
            "text",
            1,
            None,
            sort_reverse=False,
            results=results,
        )
        cached = get_search_results_cache(
            "invoice",
            "text",
            1,
            None,
            sort_reverse=False,
        )
        assert cached == results

    def test_different_query_is_separate_entry(self) -> None:
        r1 = _make_results("invoice")
        r2 = _make_results("receipt")
        set_search_results_cache(
            "invoice",
            "text",
            1,
            None,
            sort_reverse=False,
            results=r1,
        )
        set_search_results_cache(
            "receipt",
            "text",
            1,
            None,
            sort_reverse=False,
            results=r2,
        )
        assert (
            get_search_results_cache("invoice", "text", 1, None, sort_reverse=False)
            == r1
        )
        assert (
            get_search_results_cache("receipt", "text", 1, None, sort_reverse=False)
            == r2
        )

    def test_different_user_is_separate_entry(self) -> None:
        r1 = _make_results()
        r2 = _make_results()
        set_search_results_cache("q", "text", 1, None, sort_reverse=False, results=r1)
        set_search_results_cache("q", "text", 2, None, sort_reverse=False, results=r2)
        assert get_search_results_cache("q", "text", 1, None, sort_reverse=False) == r1
        assert get_search_results_cache("q", "text", 2, None, sort_reverse=False) == r2

    def test_superuser_none_is_separate_from_user(self) -> None:
        r_super = _make_results("a")
        r_user = _make_results("b")
        set_search_results_cache(
            "q",
            "text",
            None,
            None,
            sort_reverse=False,
            results=r_super,
        )
        set_search_results_cache(
            "q",
            "text",
            1,
            None,
            sort_reverse=False,
            results=r_user,
        )
        assert (
            get_search_results_cache("q", "text", None, None, sort_reverse=False)
            == r_super
        )
        assert (
            get_search_results_cache("q", "text", 1, None, sort_reverse=False) == r_user
        )

    def test_different_search_mode_is_separate_entry(self) -> None:
        r_text = _make_results()
        r_title = _make_results()
        set_search_results_cache(
            "q",
            "text",
            1,
            None,
            sort_reverse=False,
            results=r_text,
        )
        set_search_results_cache(
            "q",
            "title",
            1,
            None,
            sort_reverse=False,
            results=r_title,
        )
        assert (
            get_search_results_cache("q", "text", 1, None, sort_reverse=False) == r_text
        )
            get_search_results_cache("q", "title", 1, None, sort_reverse=False)
            == r_title
        )

    def test_bump_generation_invalidates_all_entries(self) -> None:
        results = _make_results()
        set_search_results_cache(
            "q",
            "text",
            1,
            None,
            sort_reverse=False,
            results=results,
        )
        assert (
            get_search_results_cache("q", "text", 1, None, sort_reverse=False)
            is not None
        )

        bump_search_cache_generation()

        assert (
            get_search_results_cache("q", "text", 1, None, sort_reverse=False) is None
        )

    def test_bump_generation_multiple_times(self) -> None:
        results = _make_results()
        set_search_results_cache(
            "q",
            "text",
            1,
            None,
            sort_reverse=False,
            results=results,
        )
        bump_search_cache_generation()
        bump_search_cache_generation()
        set_search_results_cache(
            "q",
            "text",
            1,
            None,
            sort_reverse=False,
            results=results,
        )
        bump_search_cache_generation()
        assert (
            get_search_results_cache("q", "text", 1, None, sort_reverse=False) is None
        )

    def test_generation_key_has_no_expiry(self) -> None:
        """Generation key must be stored without TTL so it cannot expire and silently
        reset to 0, which would make stale cache entries reachable again."""
        from django.core.cache import caches

        from documents.caching import SEARCH_GENERATION_KEY
        from documents.caching import read_cache

        bump_search_cache_generation()

        # Simulate expiry of the generation key by deleting it directly.
        read_cache.delete(SEARCH_GENERATION_KEY)

        # After expiry, bump must re-initialise without TTL.
        bump_search_cache_generation()

        # Verify the key now has no TTL (ttl returns None or -1 depending on backend).
        # Django's LocMemCache doesn't expose ttl(), so we check via the underlying
        # _expire_info dict when available; otherwise just confirm the key exists.
        raw_cache = caches["read-cache"]
        if hasattr(raw_cache, "_expire_info"):
            import time

            expire_at = raw_cache._expire_info.get(
                raw_cache.make_key(SEARCH_GENERATION_KEY),
            )
            assert expire_at is None or expire_at > time.time() + 86400 * 3650, (
                "Generation key must not have a short TTL"
            )
        else:
            # For Redis or other backends, just confirm the key is still present.
            assert read_cache.get(SEARCH_GENERATION_KEY) is not None


class TestSearchCacheIntegration:
    """Integration tests: cache is populated and invalidated via TantivyBackend."""

    def test_search_result_is_cached(self, backend: TantivyBackend) -> None:
        doc = Document.objects.create(
            title="Invoice 2024",
            content="total due",
            checksum="C1",
            pk=1,
        )
        backend.add_or_update(doc)

        # First call executes Tantivy and populates the cache.
        r1 = backend.search(
            "Invoice",
            user=None,
            page=1,
            page_size=10,
            sort_field=None,
            sort_reverse=False,
        )

        # Cache must hold the full hit list (not page-scoped).
        cached = get_search_results_cache(
            "Invoice",
            SearchMode.QUERY,
            None,
            None,
            sort_reverse=False,
        )
        assert cached is not None
        assert cached.total == r1.total
        assert cached.hits == r1.hits  # one doc, page_size=10 → same content

        # Second call must return identical results from cache.
        r2 = backend.search(
            "Invoice",
            user=None,
            page=1,
            page_size=10,
            sort_field=None,
            sort_reverse=False,
        )
        assert r1 == r2

    def test_different_pages_served_from_single_cache_entry(
        self,
        backend: TantivyBackend,
    ) -> None:
        """All pages of the same query must be served from one cache entry."""
        for i in range(1, 6):
            Document.objects.create(
                title=f"Report {i}",
                content="quarterly data",
                checksum=f"RPT{i}",
                pk=100 + i,
            )
            backend.add_or_update(Document.objects.get(pk=100 + i))

        # Page 1 populates cache.
        p1 = backend.search(
            "Report",
            user=None,
            page=1,
            page_size=2,
            sort_field=None,
            sort_reverse=False,
        )
        # Page 2 must be served from the same cache entry.
        p2 = backend.search(
            "Report",
            user=None,
            page=2,
            page_size=2,
            sort_field=None,
            sort_reverse=False,
        )

        # Results must be different slices, not the same page repeated.
        assert p1.total == p2.total == 5
        assert len(p1.hits) == 2
        assert len(p2.hits) == 2
        p1_ids = {h["id"] for h in p1.hits}
        p2_ids = {h["id"] for h in p2.hits}
        assert p1_ids.isdisjoint(p2_ids), "page 1 and page 2 must not overlap"

    def test_add_or_update_invalidates_cache(self, backend: TantivyBackend) -> None:
        doc = Document.objects.create(
            title="Old Title",
            content="content",
            checksum="C2",
            pk=2,
        )
        backend.add_or_update(doc)
        backend.search(
            "Old Title",
            user=None,
            page=1,
            page_size=10,
            sort_field=None,
            sort_reverse=False,
        )

        # Mutate the document — this must invalidate the cache.
        doc.title = "New Title"
        doc.checksum = "C2b"
        doc.save(update_fields=["title", "checksum"])
        backend.add_or_update(doc)

        # The cached entry for the old query must be gone.
        assert (
            get_search_results_cache(
                "Old Title",
                SearchMode.TEXT,
                None,
                None,
                sort_reverse=False,
            )
            is None
        )
        assert (
            get_search_results_cache(
                "Old Title",
                SearchMode.QUERY,
                None,
                None,
                sort_reverse=False,
            )
            is None
        )

    def test_remove_invalidates_cache(self, backend: TantivyBackend) -> None:
        doc = Document.objects.create(
            title="To Remove",
            content="bye",
            checksum="C3",
            pk=3,
        )
        backend.add_or_update(doc)
        backend.search(
            "To Remove",
            user=None,
            page=1,
            page_size=10,
            sort_field=None,
            sort_reverse=False,
        )

        backend.remove(doc.pk)

        assert (
            get_search_results_cache(
                "To Remove",
                SearchMode.QUERY,
                None,
                None,
                sort_reverse=False,
            )
            is None
        )

    def test_cache_is_per_user(self, backend: TantivyBackend) -> None:
        doc = Document.objects.create(
            title="Shared Doc",
            content="data",
            checksum="C4",
            pk=4,
        )
        backend.add_or_update(doc)

        user = User.objects.create_user(username="alice", pk=99)
        r_super = backend.search(
            "Shared",
            user=None,
            page=1,
            page_size=10,
            sort_field=None,
            sort_reverse=False,
        )
        r_user = backend.search(
            "Shared",
            user=user,
            page=1,
            page_size=10,
            sort_field=None,
            sort_reverse=False,
        )

        # Both should be cached but under different keys.
        assert (
            get_search_results_cache(
                "Shared",
                SearchMode.QUERY,
                None,
                None,
                sort_reverse=False,
            )
            == r_super
        )
        assert (
            get_search_results_cache(
                "Shared",
                SearchMode.QUERY,
                user.pk,
                None,
                sort_reverse=False,
            )
            == r_user
        )
