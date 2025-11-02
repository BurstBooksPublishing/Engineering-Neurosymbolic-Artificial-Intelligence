class Script:
    def __init__(self, events):
        self.events = events
        self.current_event = 0

    def next_event(self):
        if self.current_event < len(self.events) - 1:
            self.current_event += 1
            return self.events[self.current_event]
        return None

# Example script for a restaurant visit
restaurant_script = Script(["enter", "order", "eat", "pay"])

# Simulated input processing
input_process = "customer has eaten"

if input_process == "customer has eaten":
    print(restaurant_script.next_event())  # Outputs: 'pay'