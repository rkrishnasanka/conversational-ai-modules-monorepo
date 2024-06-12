# def primary_workflow(): # Change this to something more appropriate _workflow suffix

#     # Put the linearized code here and then call the functions in the order you want

#     pass


from typing import Dict, List, Tuple
import re
from nlqs.database.sqlite import retrieve_descriptions_and_types_from_db, execute_query, validate_query
from nlqs.query import get_chroma_instance,summarize, generate_query, similarity_search

data_vectors = get_chroma_instance()
column_descriptions, numerical_columns, categorical_columns = retrieve_descriptions_and_types_from_db()

def main_workflow(user_input:str, chat_history:List[Tuple[str, str]], column_descriptions_dict:Dict[str,str]=column_descriptions, numerical_columns_list:List[str,str]=numerical_columns, categorical_columns_list:List[str,str]=categorical_columns) -> Tuple[str,List[Tuple[str, str]]]:
    """This function is where the whole interaction happens. 
    It takes the user input and chat history as input and returns the response if the user's intent is either phatic_communication, profanity or sql_injection. 
    Else it returns the query result or search similarity result and the updated chat history.

    Args:
        user_input (str): The user's input.
        chat_history (list[(str, str)]): The chat history.
        column_descriptions_dict (dict[str, str]): The column descriptions.
        numerical_columns_list (list[str]): The numerical columns.
        categorical_columns_list (list[str]): The categorical columns.

    Returns:
        Tuple[str,List[Tuple[str, str]]]: The response and the updated chat history.
    """

    if not user_input.strip():
        response = ""

    user_input = re.sub(r"{|}", "", user_input)
    summarized_input = summarize(user_input, chat_history, column_descriptions_dict, numerical_columns_list, categorical_columns_list)

    if not summarized_input:
        summarized_input = summarize(user_input, chat_history, column_descriptions_dict, numerical_columns_list, categorical_columns_list)

    if not summarized_input:
        response = ""

    intent = summarized_input.user_intent
    
    if intent == "phatic_communication" or intent == "sql_injection" or intent == "profanity":
        response = ""
    
    else:
        if summarized_input.user_requested_columns:
            genenerted_query = generate_query(user_input, summarized_input, chat_history, numerical_columns_list, categorical_columns_list, column_descriptions_dict)
            if validate_query(genenerted_query):            
                query_result = execute_query(genenerted_query)
                if query_result == str([]):
                    query_result = similarity_search(data_vectors, user_input)
                response = query_result
            else:
                response = "error while generating query. Please try again."
        else:
            response = ""

    chat_history.append((user_input,response))
    return response, chat_history