from __future__ import annotations

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from app.core.config import get_settings


class AzureSearchIndexer:
    def __init__(self) -> None:
        settings = get_settings()
        credential = AzureKeyCredential(settings.azure_search_api_key)
        self.index_name = settings.azure_search_index_name
        self.index_client = SearchIndexClient(endpoint=settings.azure_search_endpoint, credential=credential)
        self.search_client = SearchClient(
            endpoint=settings.azure_search_endpoint,
            index_name=self.index_name,
            credential=credential,
        )
        self.vector_dimensions = settings.azure_search_vector_dimensions

    def build_index_schema(self) -> SearchIndex:
        fields = [
            SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True, filterable=True),
            SimpleField(name="case_id", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SearchableField(name="title", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SearchableField(name="court_name", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SearchableField(name="judge_name", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="status", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="filing_date", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SimpleField(name="judgment_date", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SimpleField(name="page_number", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
            SimpleField(name="chunk_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SearchableField(name="chunk_text", type=SearchFieldDataType.String),
            SearchField(
                name="embedding_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.vector_dimensions,
                vector_search_profile_name="legal-rag-vector-profile",
            ),
        ]
        return SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(name="legal-rag-hnsw")],
                profiles=[VectorSearchProfile(name="legal-rag-vector-profile", algorithm_configuration_name="legal-rag-hnsw")],
            ),
            semantic_search=SemanticSearch(
                configurations=[
                    SemanticConfiguration(
                        name="legal-rag-semantic-config",
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=SemanticField(field_name="title"),
                            content_fields=[SemanticField(field_name="chunk_text")],
                        ),
                    )
                ]
            ),
        )

    def create_or_update_index(self) -> None:
        self.index_client.create_or_update_index(self.build_index_schema())

    def upload_documents(self, documents: list[dict]) -> None:
        if documents:
            self.search_client.upload_documents(documents)

    def text_search(self, query: str, top_k: int, filter_expression: str | None = None) -> list[dict]:
        results = self.search_client.search(
            search_text=query,
            select=[
                "chunk_id",
                "case_id",
                "title",
                "court_name",
                "judge_name",
                "status",
                "filing_date",
                "judgment_date",
                "page_number",
                "chunk_type",
                "chunk_text",
            ],
            query_type="semantic",
            semantic_configuration_name="legal-rag-semantic-config",
            filter=filter_expression,
            top=top_k,
        )
        return [
            {
                **doc,
                "score": doc.get("@search.score", 0.0),
                "reranker_score": doc.get("@search.reranker_score"),
            }
            for doc in results
        ]

    def hybrid_search(self, query: str, embedding: list[float], top_k: int) -> list[dict]:
        vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=top_k, fields="embedding_vector")
        results = self.search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            select=[
                "chunk_id",
                "case_id",
                "title",
                "court_name",
                "judge_name",
                "status",
                "filing_date",
                "judgment_date",
                "page_number",
                "chunk_type",
                "chunk_text",
            ],
            query_type="semantic",
            semantic_configuration_name="legal-rag-semantic-config",
            top=top_k,
        )
        return [
            {
                **doc,
                "score": doc.get("@search.score", 0.0),
                "reranker_score": doc.get("@search.reranker_score"),
            }
            for doc in results
        ]
