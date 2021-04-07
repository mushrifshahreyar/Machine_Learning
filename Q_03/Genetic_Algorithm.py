import numpy as np
from matplotlib import pyplot as plt

bit_length = 6

function = lambda x : pow(x,3) + 9

def create_individual():
    individual = np.array([])
    k = np.random.randint(low=2,high=4)
    individual = np.zeros((1,6-k))
    individual = np.append(individual,np.ones((1,k)))
    np.random.shuffle(individual)

    return individual

def population_initialization():
    population = create_individual()
    for i in range(9):
        individual = create_individual()        
        population = np.vstack([population,individual])

    return population

def binary_to_decimal(individual):
    number = 0
    k = 5
    for i in range(6):
        number = number + (individual[i] * pow(2,k-i))
    return number

def fitness_calculation(population):
    
    Y = np.array([])
    _sum = 0
    for i in range(10):
        number = binary_to_decimal(population[i])
        Y = np.append(Y,function(number))
        _sum = _sum + function(number)
    
    fitness = []
    fitness[:] = Y[:] / _sum

    return fitness

def parent_selection(population, fitness):
    temp_population = []
    for i in range(10):
        temp_population.append(binary_to_decimal(population[i]))
    
    # print(fitness)
    # print(temp_population)
    parent = np.random.choice(temp_population,size=10,p=fitness)
    # print(parent)
    return parent

def crossover(parent):
    k = np.random.randint(low=1,high=6)
    
    x = np.array(list(np.binary_repr(int(parent[0]),width=6)), dtype=float)
    y = np.array(list(np.binary_repr(int(parent[9]),width=6)), dtype=float)
    
    children = np.array(x[:k])
    children = np.append(children,y[k:])

    temp = np.array(x[k:])
    temp = np.append(temp,y[:k])
        
    children = np.vstack([children,temp])

    for i in range(1,5):
        
        x = np.array([list(np.binary_repr(int(parent[i]),width=6))], dtype=float)
        y = np.array([list(np.binary_repr(int(parent[9-i]),width=6))], dtype=float)

        k = np.random.randint(low=1,high=6)
        temp = np.array(x[:k])
        temp = np.append(temp,y[k:])
        
        children = np.vstack([children,temp])

        temp = np.array(x[k:])
        temp = np.append(temp,y[:k])
        
        children = np.vstack([children,temp])
    
    return children

def mutation(children):
    k = np.random.randint(low=1,high=10)
    i = np.random.randint(low=0,high=6)
    if(children[k][i] == 1):
        children[k][i] = 0
    else:
        children[k][i] = 1
    
    
def survivor_selection(population,chilldren):
    temp_children = []
    for i in range(10):
        temp_children.append(binary_to_decimal(children[i]))
    
    temp_population = []
    for i in range(10):
        temp_population.append(binary_to_decimal(population[i]))
    
    sort_children = sorted(temp_children)
    sort_population = sorted(temp_population)

    population_new = np.array(list(np.binary_repr(int(sort_children[9]),width=6)), dtype=float)

    for i in range(1,8):
        population_new = np.vstack([population_new,np.array(list(np.binary_repr(int(sort_children[9-i]),width=6)), dtype=float)])
    
    for i in range(8,10):
        population_new = np.vstack([population_new,np.array(list(np.binary_repr(int(sort_population[i]),width=6)), dtype=float)])
    
    return population_new

def average_fitness(population):
    temp_population = []
    _sum = 0
    for i in range(10):
        number = binary_to_decimal(population[i])
        _sum = _sum + function(number)

    _sum = _sum/10 
    return _sum
    

if(__name__ == "__main__"):
    population = population_initialization()
    i = 0
    avg_fitness = []
    itr = []
    result = False
    while(True):
        fitness = fitness_calculation(population)
        avg_fitness.append(average_fitness(population))
        # print(fitness)
        parent = parent_selection(population, fitness)
        print(parent)
        itr.append(i)
        for x in parent:
            if(x == 63):
                result = True
                
        if(result):
            break        
        children = crossover(parent)
        if(i % 100 == 0):
            mutation(children)
        population = survivor_selection(population,children)
        
        i = i + 1

    print(i)
    plt.plot(itr,avg_fitness)
    plt.show()
