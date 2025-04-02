from abc import ABC

from tog.llms.base_llm import BaseLLM
from tog.models.kg import KnowledgeGraph
from tog.models.path import TopNPaths
from tog.pipeline.entity_explorer import EntityExplorer
from tog.pipeline.relation_explorer import RelationExplorer
from tog.utils.logger import setup_logger


class ExplorationLoop(ABC):
    """
    Abstract base class for the exploration loop in the pipeline.
    """

    def __init__(self, query: str,
                 llm: BaseLLM,
                 kg: KnowledgeGraph,
                 relation_explorer: RelationExplorer,
                 entity_explorer: EntityExplorer,
                 topn_paths: TopNPaths):
        """
        Initialize the exploration loop with a language model, knowledge graph, and query."
        """
        self.query = query
        self.llm = llm
        self.kg = kg
        self.relation_explorer = relation_explorer
        self.entity_explorer = entity_explorer
        self.topn_paths = topn_paths
        self.logger = setup_logger(name="exploration_loop", log_filename="exploration_loop.log")
        self.logger.info(f"ExplorationLoop initialized with query: {query}")
        self.run_loop()
    
    def run_loop(self):
        """
        Run the exploration loop.
        """
        self.logger.info("Running exploration loop")
        for path in TopNPaths.get_paths(self.query):
            self.logger.debug(f"Exploring path: {path}")
            self.relation_explorer.explore_relations(path)
            self.entity_explorer.get_entities(path)

