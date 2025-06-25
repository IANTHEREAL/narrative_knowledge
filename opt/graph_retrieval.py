import hashlib
import logging
from sqlalchemy.orm import Session
from sqlalchemy import text

from utils.uuid_utils import validate_uuid_list

logger = logging.getLogger(__name__)


def query_entities_by_ids(db: Session, entities_id: list[str]):
    # Validate input IDs to ensure they are proper UUIDs
    valid_ids = validate_uuid_list(entities_id)

    if not valid_ids:
        logger.warning("No valid UUIDs provided for entity query")
        return {}

    sql = text(
        f"""SELECT id, name, description, attributes from entities where id in :entities_id"""
    )
    res = db.execute(sql, {"entities_id": valid_ids})
    entities = {}

    try:
        for row in res.fetchall():
            entities[row.id] = {
                "id": row.id,
                "name": row.name,
                "description": row.description,
                "attributes": row.attributes,
            }
    except Exception as e:
        logger.error(f"Failed to query entities: {e}")
        return {}

    return entities


def get_relationship_by_entity_ids(db: Session, entity_ids: list[str]):
    # Validate input IDs to ensure they are proper UUIDs
    valid_ids = validate_uuid_list(entity_ids)

    if not valid_ids:
        logger.warning("No valid UUIDs provided for relationship query by entity IDs")
        return {}

    sql = text(
        f"""SELECT rel.id, source_entity.name as source_entity_name, target_entity.name as target_entity_name, rel.relationship_desc, rel.attributes
               FROM relationships as rel
               LEFT JOIN entities as source_entity ON rel.source_entity_id = source_entity.id
               LEFT JOIN entities as target_entity ON rel.target_entity_id = target_entity.id
        where rel.source_entity_id in :entity_ids or rel.target_entity_id in :entity_ids """
    )
    res = db.execute(sql, {"entity_ids": valid_ids})
    background_relationships = {}

    try:
        for row in res.fetchall():
            background_relationships[row.id] = {
                "id": row.id,
                "source_entity_name": row.source_entity_name,
                "target_entity_name": row.target_entity_name,
                "relationship_desc": row.relationship_desc,
                "attributes": row.attributes,
            }
    except Exception as e:
        logger.error(f"Failed to get relationships: {e}")
        return {}

    return background_relationships


def get_relationship_by_ids(db: Session, relationship_ids: list[str]):
    # Validate input IDs to ensure they are proper UUIDs
    valid_ids = validate_uuid_list(relationship_ids)

    if not valid_ids:
        logger.warning("No valid UUIDs provided for relationship query by IDs")
        return {}

    sql = text(
        f"""
        SELECT 
            rel.id, 
            source_entity.name as source_entity_name,
            source_entity.id as source_entity_id,
            target_entity.name as target_entity_name,
            target_entity.id as target_entity_id,
            rel.relationship_desc, rel.attributes
        FROM relationships as rel
        LEFT JOIN entities as source_entity ON rel.source_entity_id = source_entity.id
        LEFT JOIN entities as target_entity ON rel.target_entity_id = target_entity.id
        where rel.id in :relationship_ids
    """
    )
    res = db.execute(sql, {"relationship_ids": valid_ids})
    background_relationships = {}

    try:
        for row in res.fetchall():
            background_relationships[row.id] = {
                "id": row.id,
                "source_entity_name": row.source_entity_name,
                "target_entity_name": row.target_entity_name,
                "source_entity_id": row.source_entity_id,
                "target_entity_id": row.target_entity_id,
                "relationship_desc": row.relationship_desc,
                "attributes": row.attributes,
            }
    except Exception as e:
        logger.error(f"Failed to get relationships: {e}")
        return {}

    return background_relationships


def get_source_data_by_ids(db: Session, source_data_ids: list[str]):
    # Validate input IDs to ensure they are proper UUIDs
    valid_ids = validate_uuid_list(source_data_ids)

    if not valid_ids:
        logger.warning("No valid UUIDs provided for source data query by IDs")
        return {}

    sql = text(
        f"""
        SELECT id, name, content, link, source_type, attributes from source_data where id in :source_data_ids
    """
    )
    res = db.execute(sql, {"source_data_ids": valid_ids})
    source_data = {}

    try:
        for row in res.fetchall():
            source_data[row.id] = {
                "id": row.id,
                "name": row.name,
                "content": row.content,
                "link": row.link,
                "source_type": row.source_type,
                "attributes": row.attributes,
            }
    except Exception as e:
        logger.error(f"Failed to get source data: {e}")
        return {}

    return source_data


def get_source_data_by_entity_ids(db: Session, entity_ids: list[str]):
    # Validate input IDs to ensure they are proper UUIDs
    valid_ids = validate_uuid_list(entity_ids)

    if not valid_ids:
        logger.warning("No valid UUIDs provided for source data query by entity IDs")
        return []

    sql = text(
        f"""
        SELECT 
            sd.id, 
            sd.name, 
            sd.content, 
            sd.link, 
            sd.source_type, 
            sd.attributes
        FROM source_data as sd
        INNER JOIN source_graph_mapping as sgm ON sd.id = sgm.source_id
        WHERE sgm.graph_element_type = 'entity' 
        AND sgm.graph_element_id IN :entity_ids
    """
    )
    res = db.execute(sql, {"entity_ids": valid_ids})
    source_data = {}

    try:
        for row in res.fetchall():
            content_hash = hashlib.sha256(row.content.encode()).hexdigest()
            source_data[content_hash] = {
                "id": row.id,
                "name": row.name,
                "content": row.content,
                "link": row.link,
                "source_type": row.source_type,
                "attributes": row.attributes,
            }
    except Exception as e:
        logger.error(f"Failed to get source data by entity ids: {e}")
        return {}

    return list(source_data.values())


def get_source_data_by_relationship_ids(db: Session, relationship_ids: list[str]):
    # Validate input IDs to ensure they are proper UUIDs
    valid_ids = validate_uuid_list(relationship_ids)

    if not valid_ids:
        logger.warning(
            "No valid UUIDs provided for source data query by relationship IDs"
        )
        return []

    sql = text(
        f"""
        SELECT 
            sd.id, 
            sd.name, 
            sd.content, 
            sd.link, 
            sd.source_type, 
            sd.attributes
        FROM source_data as sd
        INNER JOIN source_graph_mapping as sgm ON sd.id = sgm.source_id
        WHERE sgm.graph_element_type = 'relationship' 
        AND sgm.graph_element_id IN :relationship_ids
    """
    )
    res = db.execute(sql, {"relationship_ids": valid_ids})
    source_data = {}

    try:
        for row in res.fetchall():
            content_hash = hashlib.sha256(row.content.encode()).hexdigest()
            source_data[content_hash] = {
                "id": row.id,
                "name": row.name,
                "content": row.content,
                "link": row.link,
                "source_type": row.source_type,
                "attributes": row.attributes,
            }
    except Exception as e:
        logger.error(f"Failed to get source data by relationship ids: {e}")
        return {}

    return list(source_data.values())
