def check_constraints(output):
    assert output in ["Action A", "Action B"], "Unexpected action generated"
    return output

@monitor
def decision_process(data):
    processed = neural_process(data)
    decision = symbolic_reasoning(processed["processed_data"])
    return check_constraints(decision)