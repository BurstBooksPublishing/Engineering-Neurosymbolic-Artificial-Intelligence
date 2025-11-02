def predict_outcomes(tx, condition):
    query = """
    MATCH (n:H2O)-[r:REACTS_WITH]->(m)
    WHERE r.condition = $condition
    RETURN m.name AS product
    """
    results = tx.run(query, condition=condition)
    products = [record["product"] for record in results]

    tensor_condition = torch.tensor([float(condition)])
    predictions = model(tensor_condition)

    return dict(zip(products, predictions.tolist()))

with driver.session() as session:
    outcomes = session.read_transaction(predict_outcomes, 'Electrolysis')
    print(outcomes)