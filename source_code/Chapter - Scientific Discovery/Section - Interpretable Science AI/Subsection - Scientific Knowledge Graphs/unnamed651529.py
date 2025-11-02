from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

def create_graph(tx):
    tx.run("CREATE (water:H2O {name: 'Water'})")
    tx.run("CREATE (oxygen:O2 {name: 'Oxygen'})")
    tx.run("CREATE (hydrogen:H2 {name: 'Hydrogen'})")
    tx.run("CREATE (water)-[:REACTS_WITH {condition: 'Electrolysis'}]->(oxygen)")
    tx.run("CREATE (water)-[:REACTS_WITH {condition: 'Electrolysis'}]->(hydrogen)")

with driver.session() as session:
    session.write_transaction(create_graph)