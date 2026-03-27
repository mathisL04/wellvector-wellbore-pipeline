"""Pydantic models, constants, and Norwegian→English glossary for casing extraction."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Norwegian → English glossary for well-document terminology
# ---------------------------------------------------------------------------

NORWEGIAN_GLOSSARY: dict[str, str] = {
    "foringsrør": "casing",
    "foringsrørstørrelse": "casing size",
    "foringsrørdybde": "casing depth",
    "hulstørrelse": "hole size",
    "hulldiameter": "hole diameter",
    "borehull": "borehole",
    "brønndesign": "well design",
    "brønnstigevekt": "mud weight",
    "sementering": "cementation",
    "sementasjon": "cementation",
    "lekkasjetest": "leak-off test",
    "formasjonsintegritetstest": "formation integrity test",
    "borekrone": "drill bit",
    "lederør": "conductor casing",
    "overflaterør": "surface casing",
    "mellomrør": "intermediate casing",
    "produksjonsrør": "production casing",
    "fôringssko": "casing shoe",
    "sjøbunn": "seabed",
    "rotasjonsbord": "rotary table",
    "borevæske": "drilling fluid",
    "borevæskevekt": "mud weight",
    "ekvivalent slamvekt": "equivalent mud weight",
    "tetthet": "density",
    "dybde": "depth",
    "diameter": "diameter",
    "tommer": "inches",
}

# Keywords that strongly indicate casing design content
CASING_KEYWORDS: list[str] = [
    # English
    "casing", "casing string", "casing shoe", "conductor",
    "surface casing", "intermediate casing", "production casing",
    "liner", "hole size", "hole section", "bit size",
    "LOT", "FIT", "leak-off", "leak off", "formation integrity",
    "mud weight", "mud eqv", "EMW",
    "well design", "well completion", "completion report",
    "wellbore diagram", "wellbore schematic",
    "csg", "csg shoe",
    # Norwegian
    "foringsrør", "hulstørrelse", "lekkasjetest",
    "formasjonsintegritetstest", "brønndesign",
    "lederør", "overflaterør", "mellomrør", "produksjonsrør",
    "fôringssko", "borekrone",
]

# Document names that are very likely to contain casing data
HIGH_PRIORITY_DOC_PATTERNS: list[str] = [
    "WELL_COMPLETION_REPORT",
    "COMPLETION_REPORT",
    "Completion_Report",
    "completion_log",
    "WDSS",
    "INDIVIDUAL_WELL_RECORD",
    "DRILLING_FLUID_SUMMARY",
    "FORMATION_TEST",
    "AAODC_REPORT",
    "EXPLORATORY_TEST",
    "CHANGE_IN_DRILLING_PROGRAM",
]

# Document names that are almost certainly irrelevant
LOW_PRIORITY_DOC_PATTERNS: list[str] = [
    "GEOCHEMICAL", "GCH", "LITHOLOGY", "LITHOLOGIC",
    "PALEOCENE", "PALAEONTOL", "PALYNOLOG",
    "CORE_ANALYSIS", "CORE_DESCRIPTION", "CORE_LOG",
    "CORE_LABORATORIES", "SIDEWALL_CORE",
    "SEDIMENTOLOGY", "MICROPALAEONTOL",
    "DRILL_STEM_TEST", "DST_NO", "WELL_PRODUCTION_TEST",
    "VELOCITY_SURVEY", "VELOG_PROCESSING",
    "SEISMOIGRAM", "DIPMETER",
    "RAPPORT_GEOCHIMIQUE", "SOURCE_ROCK",
    "RESERVOIR_FLUID", "WET_SAMPLES",
    "CALCINITY", "BAILED_SAMPLE",
    "BIOSTRATIGRAPHY", "CLASTIC_SEDIMENTS",
    "COMPLEX_LITHOLOGY", "COD",
    "GAS_LIQUID", "LOGGING_RAPPORT",
    "KONTINENTALSOKKELEN", "SURVEY_REPORT",
    "TEMPORARY_SUSPENSION", "SURFACE_AND_BOTTOM",
    "SAND_DISTRIBUTION", "GEOCHEM",
    "DRILLING_MUD",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CasingType(str, Enum):
    CONDUCTOR = "Conductor"
    SURFACE = "Surface"
    INTERMEDIATE = "Intermediate"
    PRODUCTION = "Production"
    LINER = "Liner"
    OTHER = "Other"


class FormationTestType(str, Enum):
    LOT = "LOT"
    FIT = "FIT"
    XLOT = "XLOT"
    UNKNOWN = "Unknown"


class DocumentRelevance(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    IRRELEVANT = "irrelevant"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class DocumentMeta(BaseModel):
    """Metadata for a single wellbore document from the CSV."""
    wellbore: str
    doc_type: str
    doc_name: str
    url: str
    format: str
    size_kb: int
    npd_id: int


class TriageResult(BaseModel):
    """Result of Stage 1 triage for a single document."""
    document: DocumentMeta
    relevance: DocumentRelevance
    relevance_score: float = Field(ge=0.0, le=1.0)
    relevant_pages: list[int] = Field(default_factory=list)
    is_scanned: bool = False
    keyword_hits: list[str] = Field(default_factory=list)


class CasingRecord(BaseModel):
    """A single casing row in the output schema."""
    wellbore: str
    casing_type: str | None = None
    casing_diameter_in: float | None = None
    casing_depth_m: float | None = None
    hole_diameter_in: float | None = None
    hole_depth_m: float | None = None
    lot_fit_mud_eqv_gcm3: float | None = None
    formation_test_type: str | None = None


class ExtractionResult(BaseModel):
    """Result of full extraction from a single document."""
    document: DocumentMeta
    records: list[CasingRecord] = Field(default_factory=list)
    raw_text_used: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    errors: list[str] = Field(default_factory=list)


class PipelineStats(BaseModel):
    """Token usage and performance stats for the full pipeline run."""
    total_documents: int = 0
    documents_downloaded: int = 0
    documents_triaged_relevant: int = 0
    documents_extracted: int = 0
    pages_processed: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_calls: int = 0
    ocr_pages: int = 0
