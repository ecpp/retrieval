# Import main classes
from .image_encoder import ImageEncoder
from .vector_database import VectorDatabase
from .data_processor import DataProcessor
from .evaluator import RetrievalEvaluator
from .retrieval_system import RetrievalSystem

# Import metadata components
try:
    from .metadata_encoder import MetadataEncoder
    from .fusion_module import FusionModule
except ImportError:
    pass  # These will be handled in the retrieval_system

# Version
__version__ = '0.1.0'
