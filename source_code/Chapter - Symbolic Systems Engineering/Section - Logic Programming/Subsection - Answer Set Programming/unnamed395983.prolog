% Define rules
object(X, red) :- detected(X, red).
object(X, blue) :- detected(X, blue).

% Define a higher-level inference
conflict(X, Y) :- object(X, red), object(Y, blue), next_to(X, Y).

% Query for conflicts
#show conflict/2.