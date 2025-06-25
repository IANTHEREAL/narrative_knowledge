#!/usr/bin/env python3
"""
Simple script to export knowledge graph entities and relationships to JSON files.
"""

import os
import json
import logging
from knowledge_graph.models import Entity, Relationship
from setting.db import db_manager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Get database URI from environment variable
    database_uri = os.getenv("GRAPH_DATABASE_URI")
    if not database_uri:
        logger.error("Please set GRAPH_DATABASE_URI environment variable")
        return
    
    # Connect to database
    session_factory = db_manager.get_session_factory(database_uri)
    
    with session_factory() as db:
        logger.info("Connected to database, starting export...")
        
        # Export entities
        logger.info("Exporting entities...")
        entities = db.query(Entity).all()
        entities_data = []
        for entity in entities:
            entities_data.append({
                "id": entity.id,
                "name": entity.name,
                "description": entity.description,
                "attributes": entity.attributes or {}
            })
        
        with open("entities.json", "w", encoding="utf-8") as f:
            json.dump(entities_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Exported {len(entities_data)} entities to entities.json")
        
        # Export relationships
        logger.info("Exporting relationships...")
        relationships = db.query(Relationship).all()
        relationships_data = []
        for rel in relationships:
            relationships_data.append({
                "source_entity_id": rel.source_entity_id,
                "target_entity_id": rel.target_entity_id,
                "description": rel.relationship_desc,
                "attributes": rel.attributes or {}
            })
        
        with open("relationships.json", "w", encoding="utf-8") as f:
            json.dump(relationships_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Exported {len(relationships_data)} relationships to relationships.json")
        
        logger.info("Export completed successfully!")

if __name__ == "__main__":
    main() 