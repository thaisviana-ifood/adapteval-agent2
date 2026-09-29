"""Structured output schema for the thesis evaluation pipeline

Mirrors the shape requested for the pipeline's final output:
Avaliação: <métricas, valores, confiança>, nota de confiança global,
relatório de consolidação das métricas.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class MetricaAvaliada(BaseModel):
    """A single (metric, value, confidence) triple"""

    metrica: str = Field(description="Nome da métrica avaliada")
    valor: float = Field(
        description=(
            "Valor obtido para a métrica: 1.0/0.0 para vereditos de rubrica "
            "booleana (ex: jury.*, <evaluator>.*), ou escala 0-10 para as "
            "métricas de consolidação final (final.*)"
        )
    )
    confianca: float = Field(description="Confiança associada ao valor (0-1)")


class RelatorioConsolidacao(BaseModel):
    """Consolidated metrics report, produced by the metrics_aggregation node"""

    evaluation_id: str
    final_score: float
    confidence: float
    recommendation: str
    summary: Dict[str, List[str]]
    detailed_breakdown: Dict[str, Any]


class EvaluationOutput(BaseModel):
    """Final structured output of the evaluation pipeline"""

    avaliacao: List[MetricaAvaliada]
    nota_confianca_global: float
    relatorio_consolidacao_metricas: RelatorioConsolidacao
