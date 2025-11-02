from pyDatalog import pyDatalog

pyDatalog.create_terms('X, Y, COLOR, SHAPE, on_table, color_of, shape_of')

# Define facts
on_table('blue_ball')
on_table('red_cube')

color_of('blue_ball', 'blue')
color_of('red_cube', 'red')

shape_of('blue_ball', 'ball')
shape_of('red_cube', 'cube')

# Query: What objects are on the table?
print(pyDatalog.ask('on_table(X)'))

# Query: What color are the objects on the table?
print(pyDatalog.ask('color_of(X, COLOR) & on_table(X)'))