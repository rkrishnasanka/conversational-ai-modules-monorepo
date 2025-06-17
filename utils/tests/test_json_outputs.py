import pytest

from utils.json_outputs import validate_llm_output_keys


def test_validate_llm_output_keys_no_missing_keys():
    llm_output = {"key1": "value1", "key2": {"subkey1": "subvalue1", "subkey2": "subvalue2"}}
    reference_dict = {"key1": "value1", "key2": {"subkey1": "subvalue1", "subkey2": "subvalue2"}}
    assert validate_llm_output_keys(llm_output, reference_dict) == []


def test_validate_llm_output_keys_missing_top_level_key():
    llm_output = {"key1": "value1"}
    reference_dict = {"key1": "value1", "key2": {"subkey1": "subvalue1", "subkey2": "subvalue2"}}
    assert validate_llm_output_keys(llm_output, reference_dict) == ["key2"]


def test_validate_llm_output_keys_missing_nested_key():
    llm_output = {"key1": "value1", "key2": {"subkey1": "subvalue1"}}
    reference_dict = {"key1": "value1", "key2": {"subkey1": "subvalue1", "subkey2": "subvalue2"}}
    assert validate_llm_output_keys(llm_output, reference_dict) == ["key2.subkey2"]


def test_validate_llm_output_keys_multiple_missing_keys():
    llm_output = {"key1": "value1"}
    reference_dict = {"key1": "value1", "key2": {"subkey1": "subvalue1", "subkey2": "subvalue2"}, "key3": "value3"}
    assert validate_llm_output_keys(llm_output, reference_dict) == ["key2", "key3"]


def test_validate_llm_output_keys_no_keys_in_llm_output():
    llm_output = {}
    reference_dict = {"key1": "value1", "key2": {"subkey1": "subvalue1", "subkey2": "subvalue2"}}
    assert validate_llm_output_keys(llm_output, reference_dict) == ["key1", "key2"]


def test_validate_llm_output_keys_empty_reference_dict():
    llm_output = {"key1": "value1", "key2": {"subkey1": "subvalue1", "subkey2": "subvalue2"}}
    reference_dict = {}
    assert validate_llm_output_keys(llm_output, reference_dict) == []
