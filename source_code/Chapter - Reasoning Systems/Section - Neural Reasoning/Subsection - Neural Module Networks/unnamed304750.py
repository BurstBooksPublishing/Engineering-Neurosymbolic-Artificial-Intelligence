def parse_question(question):
    # Simplified parsing logic to determine which modules are needed
    if 'where' in question:
        return [FindModule(), DescribeModule()]
    elif 'relationship' in question:
        return [FindModule(), FindModule(), RelateModule()]
    else:
        return [FindModule()]

def answer_question(question, image):
    modules = parse_question(question)
    outputs = []
    
    for module in modules:
        if isinstance(module, RelateModule):
            # Assuming two inputs for relation module
            output = module(outputs[-2], outputs[-1])
        else:
            output = module(image)
        
        outputs.append(output)
    
    return outputs[-1]  # Return the output of the last module