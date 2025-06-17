import pytest

from nlqs.query_construction import (
    construct_descriptive_search_query_fragments,
    construct_quantitaive_search_query_fragments,
    join_fragments,
)
from nlqs.vectordb_driver import VectorDBDriver


def test_join_fragments_with_custom_joiner():
    fragments = ["fragment1", "fragment2", "fragment3"]
    joiner = "OR"
    expected = "fragment1 OR fragment2 OR fragment3"
    assert join_fragments(fragments, joiner) == expected


def test_join_fragments_with_single_fragment():
    fragments = ["fragment1"]
    expected = "fragment1"
    assert join_fragments(fragments) == expected


def test_join_fragments_with_empty_list():
    fragments = []
    expected = ""
    assert join_fragments(fragments) == expected


def test_join_fragments_with_custom_joiner_and_single_fragment():
    fragments = ["fragment1"]
    joiner = "OR"
    expected = "fragment1"
    assert join_fragments(fragments, joiner) == expected


def test_join_fragments_with_default_joiner():
    fragments = ["fragment1", "fragment2", "fragment3"]
    expected = "fragment1 AND fragment2 AND fragment3"
    assert join_fragments(fragments) == expected


def test_construct_quantitaive_search_query_fragments_with_conditions():
    quantitaive_data = {"age": ">30", "salary": "<=50000", "height": ">=180", "weight": "=70", "name": "John"}
    expected = ["age > 30", "salary <= 50000", "height >= 180", "weight = 70"]
    assert construct_quantitaive_search_query_fragments(quantitaive_data) == expected


def test_construct_quantitaive_search_query_fragments_with_empty_dict():
    quantitaive_data = {}
    expected = []
    assert construct_quantitaive_search_query_fragments(quantitaive_data) == expected


def test_construct_quantitaive_search_query_fragments_with_single_condition():
    quantitaive_data = {"age": ">30"}
    expected = ["age > 30"]
    assert construct_quantitaive_search_query_fragments(quantitaive_data) == expected


def test_construct_quantitaive_search_query_fragments_with_like_condition():
    quantitaive_data = {"name": "John"}
    expected = []
    assert construct_quantitaive_search_query_fragments(quantitaive_data) == expected


def test_construct_quantitaive_search_query_fragments_with_mixed_conditions():
    quantitaive_data = {"age": ">30", "name": "John", "salary": "<=50000"}
    expected = ["age > 30", "salary <= 50000"]
    assert construct_quantitaive_search_query_fragments(quantitaive_data) == expected


def test_construct_descriptive_search_query_fragments(vectordb_driver: VectorDBDriver):
    descriptive_data = {"Description": "creamy", "Type": "white"}
    expected_pattern = "id IN "

    result = construct_descriptive_search_query_fragments(descriptive_data, vectordb_driver)

    assert len(result["Type"]) == 0

    # Test that each of the strings for the Description are gonna have the expercted pattern in them
    for description in result["Description"]:
        assert expected_pattern in description
