from multiprocessing import Process, Manager

def neural_network_process(shared_dict, input_data):
    # Simulate neural network processing
    shared_dict['features'] = [input_data * 2]  # Placeholder for actual NN output

def symbolic_process(shared_dict):
    # Access the precomputed features
    features = shared_dict['features']
    # Simulate symbolic reasoning
    result = all(f % 4 == 0 for f in features)
    print("Symbolic output:", result)

if __name__ == "__main__":
    manager = Manager()
    shared_dict = manager.dict()
    input_data = 10

    p1 = Process(target=neural_network_process, args=(shared_dict, input_data))
    p2 = Process(target=symbolic_process, args=(shared_dict,))

    p1.start()
    p1.join()
    p2.start()
    p2.join()