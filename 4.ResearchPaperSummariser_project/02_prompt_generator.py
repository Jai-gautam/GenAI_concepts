from langchain_core.prompts import PromptTemplate,load_prompt
template = PromptTemplate(
    template = """Summarise the research paper titled '{paper_title}' in a {style_type} style and {length_type} length.""",
    input_variables = ["paper_title", "style_type", "length_type"],     )

template.save("prompt_template.json")





"""
where i have to call this template i have to call load_prompt function and write the path of json file from where 
i want to load the template and then i can use that template to generate the prompt for my model.
for example : 

template = load_prompt("prompt_template.json")
... etc



"""