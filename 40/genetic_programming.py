import random
import math
import operator
from collections import defaultdict

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not found. Using simple gradient descent for constant optimization.")


class Dimension:
    def __init__(self, mass=0, length=0, time=0, temperature=0, 
                 current=0, luminous=0, substance=0):
        self.dims = {
            'M': mass,
            'L': length,
            'T': time,
            'θ': temperature,
            'I': current,
            'J': luminous,
            'N': substance
        }
    
    def __add__(self, other):
        if not isinstance(other, Dimension):
            return NotImplemented
        return Dimension(
            mass=self.dims['M'] + other.dims['M'],
            length=self.dims['L'] + other.dims['L'],
            time=self.dims['T'] + other.dims['T'],
            temperature=self.dims['θ'] + other.dims['θ'],
            current=self.dims['I'] + other.dims['I'],
            luminous=self.dims['J'] + other.dims['J'],
            substance=self.dims['N'] + other.dims['N']
        )
    
    def __sub__(self, other):
        if not isinstance(other, Dimension):
            return NotImplemented
        return Dimension(
            mass=self.dims['M'] - other.dims['M'],
            length=self.dims['L'] - other.dims['L'],
            time=self.dims['T'] - other.dims['T'],
            temperature=self.dims['θ'] - other.dims['θ'],
            current=self.dims['I'] - other.dims['I'],
            luminous=self.dims['J'] - other.dims['J'],
            substance=self.dims['N'] - other.dims['N']
        )
    
    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            return NotImplemented
        return Dimension(
            mass=self.dims['M'] * scalar,
            length=self.dims['L'] * scalar,
            time=self.dims['T'] * scalar,
            temperature=self.dims['θ'] * scalar,
            current=self.dims['I'] * scalar,
            luminous=self.dims['J'] * scalar,
            substance=self.dims['N'] * scalar
        )
    
    def __eq__(self, other):
        if not isinstance(other, Dimension):
            return False
        return all(abs(self.dims[k] - other.dims[k]) < 1e-10 for k in self.dims)
    
    def __hash__(self):
        return hash(tuple(round(self.dims[k], 10) for k in sorted(self.dims.keys())))
    
    def is_dimensionless(self):
        return all(abs(v) < 1e-10 for v in self.dims.values())
    
    def __str__(self):
        parts = []
        for dim, exp in self.dims.items():
            if abs(exp) > 1e-10:
                if abs(exp - 1) < 1e-10:
                    parts.append(dim)
                elif abs(exp + 1) < 1e-10:
                    parts.append(f"{dim}⁻¹")
                else:
                    parts.append(f"{dim}^{exp:.1f}")
        return '·'.join(parts) if parts else '1'
    
    def __repr__(self):
        return f"Dimension({self})"


DIMENSIONLESS = Dimension()


DIMENSION_VARIABLES = {
    't': Dimension(time=1),
    'x': Dimension(length=1),
    'y': Dimension(length=1),
    'z': Dimension(length=1),
    'm': Dimension(mass=1),
    'T': Dimension(temperature=1),
    'v': Dimension(length=1, time=-1),
    'a': Dimension(length=1, time=-2),
    'F': Dimension(mass=1, length=1, time=-2),
    'E': Dimension(mass=1, length=2, time=-2),
    'P': Dimension(mass=1, length=2, time=-3),
    'ρ': Dimension(mass=1, length=-3),
    'p': Dimension(mass=1, length=-1, time=-2),
}


def get_variable_dimension(var_name):
    return DIMENSION_VARIABLES.get(var_name, DIMENSIONLESS)


class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def evaluate(self, x):
        if self.value == 'x':
            return x
        elif isinstance(self.value, (int, float)):
            return self.value
        elif self.value in OPERATORS:
            left_val = self.left.evaluate(x)
            right_val = self.right.evaluate(x)
            try:
                return OPERATORS[self.value](left_val, right_val)
            except (ZeroDivisionError, ValueError, OverflowError):
                return float('inf')
        elif self.value in FUNCTIONS:
            arg = self.left.evaluate(x)
            try:
                return FUNCTIONS[self.value](arg)
            except (ValueError, OverflowError):
                return float('inf')
        return 0

    def depth(self):
        if self.left is None and self.right is None:
            return 1
        left_depth = self.left.depth() if self.left else 0
        right_depth = self.right.depth() if self.right else 0
        return 1 + max(left_depth, right_depth)

    def size(self):
        size = 1
        if self.left:
            size += self.left.size()
        if self.right:
            size += self.right.size()
        return size

    def __str__(self):
        if self.value == 'x':
            return 'x'
        elif isinstance(self.value, (int, float)):
            return f'{self.value:.2f}' if isinstance(self.value, float) else str(self.value)
        elif self.value in OPERATORS:
            return f'({self.left} {OPERATOR_SYMBOLS[self.value]} {self.right})'
        elif self.value in FUNCTIONS:
            return f'{self.value}({self.left})'
        return str(self.value)

    def clone(self):
        left_clone = self.left.clone() if self.left else None
        right_clone = self.right.clone() if self.right else None
        return Node(self.value, left_clone, right_clone)
    
    def collect_constants(self):
        constants = []
        def collect(node):
            if isinstance(node.value, (int, float)):
                constants.append(node)
            if node.left:
                collect(node.left)
            if node.right:
                collect(node.right)
        collect(self)
        return constants
    
    def compute_dimension(self, var_dimensions=None):
        if var_dimensions is None:
            var_dimensions = {}
        
        if isinstance(self.value, (int, float)):
            return DIMENSIONLESS, True
        
        if isinstance(self.value, str) and self.value not in OPERATORS and self.value not in FUNCTIONS:
            dim = var_dimensions.get(self.value, get_variable_dimension(self.value))
            return dim, True
        
        if self.value in OPERATORS:
            left_dim, left_ok = self.left.compute_dimension(var_dimensions) if self.left else (DIMENSIONLESS, True)
            right_dim, right_ok = self.right.compute_dimension(var_dimensions) if self.right else (DIMENSIONLESS, True)
            
            if not (left_ok and right_ok):
                return DIMENSIONLESS, False
            
            op = self.value
            if op in ('+', '-'):
                if left_dim == right_dim:
                    return left_dim, True
                else:
                    return DIMENSIONLESS, False
            elif op == '*':
                return left_dim + right_dim, True
            elif op == '/':
                return left_dim - right_dim, True
        
        if self.value in FUNCTIONS:
            arg_dim, arg_ok = self.left.compute_dimension(var_dimensions) if self.left else (DIMENSIONLESS, True)
            
            if not arg_ok:
                return DIMENSIONLESS, False
            
            if self.value in ('sin', 'cos', 'exp', 'log', 'sqrt'):
                if arg_dim.is_dimensionless():
                    return DIMENSIONLESS, True
                else:
                    return DIMENSIONLESS, False
        
        return DIMENSIONLESS, True
    
    def check_dimensional_consistency(self, var_dimensions=None):
        _, is_consistent = self.compute_dimension(var_dimensions)
        return is_consistent
    
    def get_result_dimension(self, var_dimensions=None):
        dim, _ = self.compute_dimension(var_dimensions)
        return dim
    
    def count_dimensional_errors(self, var_dimensions=None):
        errors = 0
        
        if self.value in OPERATORS:
            left_dim, left_ok = self.left.compute_dimension(var_dimensions) if self.left else (DIMENSIONLESS, True)
            right_dim, right_ok = self.right.compute_dimension(var_dimensions) if self.right else (DIMENSIONLESS, True)
            
            if self.value in ('+', '-') and left_dim != right_dim:
                errors += 1
            
            if self.left:
                errors += self.left.count_dimensional_errors(var_dimensions)
            if self.right:
                errors += self.right.count_dimensional_errors(var_dimensions)
        
        elif self.value in FUNCTIONS:
            arg_dim, _ = self.left.compute_dimension(var_dimensions) if self.left else (DIMENSIONLESS, True)
            if not arg_dim.is_dimensionless():
                errors += 1
            if self.left:
                errors += self.left.count_dimensional_errors(var_dimensions)
        
        return errors


OPERATORS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
}

OPERATOR_SYMBOLS = {
    '+': '+',
    '-': '-',
    '*': '*',
    '/': '/',
}

FUNCTIONS = {
    'sin': math.sin,
    'cos': math.cos,
    'exp': math.exp,
    'log': math.log,
    'sqrt': math.sqrt,
}

TERMINALS = ['x'] + [i for i in range(-5, 6)]


def random_terminal():
    term = random.choice(TERMINALS)
    if term == 'x':
        return Node('x')
    else:
        return Node(float(term))


def random_function():
    if random.random() < 0.7:
        op = random.choice(list(OPERATORS.keys()))
        return Node(op)
    else:
        func = random.choice(list(FUNCTIONS.keys()))
        return Node(func)


def generate_full(depth):
    if depth == 0:
        return random_terminal()
    node = random_function()
    node.left = generate_full(depth - 1)
    if node.value in OPERATORS:
        node.right = generate_full(depth - 1)
    return node


def generate_grow(depth):
    if depth == 0 or random.random() < 0.3:
        return random_terminal()
    node = random_function()
    node.left = generate_grow(depth - 1)
    if node.value in OPERATORS:
        node.right = generate_grow(depth - 1)
    return node


def get_random_node(node, depth=0):
    nodes = []
    
    def collect(n, d):
        nodes.append((n, d))
        if n.left:
            collect(n.left, d + 1)
        if n.right:
            collect(n.right, d + 1)
    
    collect(node, 0)
    return random.choice(nodes)


def replace_node(parent, old_child, new_child):
    if parent.left == old_child:
        parent.left = new_child
    elif parent.right == old_child:
        parent.right = new_child


def crossover(parent1, parent2):
    child1 = parent1.clone()
    child2 = parent2.clone()
    
    node1, depth1 = get_random_node(child1)
    node2, depth2 = get_random_node(child2)
    
    node1.value, node2.value = node2.value, node1.value
    node1.left, node2.left = node2.left, node1.left
    node1.right, node2.right = node2.right, node1.right
    
    return child1, child2


def mutate(node, max_depth=6):
    mutated = node.clone()
    target_node, depth = get_random_node(mutated)
    
    remaining_depth = max(0, max_depth - depth)
    new_subtree = generate_grow(remaining_depth) if remaining_depth > 0 else random_terminal()
    
    target_node.value = new_subtree.value
    target_node.left = new_subtree.left
    target_node.right = new_subtree.right
    
    return mutated


def tournament_selection(population, fitnesses, tournament_size=3):
    tournament_indices = random.sample(range(len(population)), tournament_size)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    best_idx = tournament_indices[tournament_fitnesses.index(min(tournament_fitnesses))]
    return population[best_idx]


def simplify_expression(node):
    if node is None:
        return None
    
    node.left = simplify_expression(node.left)
    node.right = simplify_expression(node.right)
    
    if node.value == '+':
        if node.left and isinstance(node.left.value, (int, float)) and node.left.value == 0:
            return node.right.clone() if node.right else None
        if node.right and isinstance(node.right.value, (int, float)) and node.right.value == 0:
            return node.left.clone() if node.left else None
        if (node.left and isinstance(node.left.value, (int, float)) and 
            node.right and isinstance(node.right.value, (int, float))):
            return Node(node.left.value + node.right.value)
    
    elif node.value == '-':
        if node.right and isinstance(node.right.value, (int, float)) and node.right.value == 0:
            return node.left.clone() if node.left else None
        if (node.left and isinstance(node.left.value, (int, float)) and 
            node.right and isinstance(node.right.value, (int, float))):
            return Node(node.left.value - node.right.value)
    
    elif node.value == '*':
        if node.left and isinstance(node.left.value, (int, float)) and node.left.value == 0:
            return Node(0.0)
        if node.right and isinstance(node.right.value, (int, float)) and node.right.value == 0:
            return Node(0.0)
        if node.left and isinstance(node.left.value, (int, float)) and node.left.value == 1:
            return node.right.clone() if node.right else None
        if node.right and isinstance(node.right.value, (int, float)) and node.right.value == 1:
            return node.left.clone() if node.left else None
        if (node.left and isinstance(node.left.value, (int, float)) and 
            node.right and isinstance(node.right.value, (int, float))):
            return Node(node.left.value * node.right.value)
    
    elif node.value == '/':
        if node.left and isinstance(node.left.value, (int, float)) and node.left.value == 0:
            return Node(0.0)
        if node.right and isinstance(node.right.value, (int, float)) and node.right.value == 1:
            return node.left.clone() if node.left else None
        if (node.left and isinstance(node.left.value, (int, float)) and 
            node.right and isinstance(node.right.value, (int, float)) and 
            node.right.value != 0):
            return Node(node.left.value / node.right.value)
    
    return node


def calculate_mse(individual, x_data, y_data):
    total_error = 0.0
    for x, y_target in zip(x_data, y_data):
        y_pred = individual.evaluate(x)
        if y_pred == float('inf') or abs(y_pred) > 1e10:
            return float('inf')
        total_error += (y_pred - y_target) ** 2
    return total_error / len(x_data)


def check_dimensional_correctness(expression, target_dimension=None, var_dimensions=None):
    if var_dimensions is None:
        var_dimensions = {}
    
    is_consistent = expression.check_dimensional_consistency(var_dimensions)
    
    if not is_consistent:
        return False, None
    
    result_dim = expression.get_result_dimension(var_dimensions)
    
    if target_dimension is not None:
        return result_dim == target_dimension, result_dim
    
    return True, result_dim


def calculate_dimensional_penalty(individual, target_dimension=None, var_dimensions=None):
    if var_dimensions is None:
        var_dimensions = {}
    
    num_errors = individual.count_dimensional_errors(var_dimensions)
    
    result_dim = individual.get_result_dimension(var_dimensions)
    
    if target_dimension is not None and result_dim != target_dimension:
        num_errors += 1
    
    penalty = num_errors * 1000.0
    
    return penalty, num_errors, result_dim


def calculate_fitness(individual, x_data, y_data, target_dimension=None, var_dimensions=None, use_dimensional_check=True):
    mse = calculate_mse(individual, x_data, y_data)
    
    if mse == float('inf'):
        return float('inf')
    
    size_penalty = individual.size() * 0.001
    
    dimensional_penalty = 0.0
    if use_dimensional_check:
        dim_penalty, _, _ = calculate_dimensional_penalty(
            individual, target_dimension, var_dimensions
        )
        dimensional_penalty = dim_penalty
    
    return mse + size_penalty + dimensional_penalty


def gradient_descent_optimize(individual, x_data, y_data, constant_nodes, max_iterations=50):
    learning_rate = 0.01
    epsilon = 1e-6
    
    for _ in range(max_iterations):
        gradients = []
        current_mse = calculate_mse(individual, x_data, y_data)
        
        if current_mse == float('inf'):
            break
        
        for i, node in enumerate(constant_nodes):
            original_value = node.value
            
            node.value = original_value + epsilon
            mse_plus = calculate_mse(individual, x_data, y_data)
            
            node.value = original_value - epsilon
            mse_minus = calculate_mse(individual, x_data, y_data)
            
            node.value = original_value
            
            if mse_plus == float('inf') or mse_minus == float('inf'):
                gradients.append(0.0)
            else:
                gradient = (mse_plus - mse_minus) / (2 * epsilon)
                gradients.append(gradient)
        
        for node, grad in zip(constant_nodes, gradients):
            node.value -= learning_rate * grad
            node.value = max(-100, min(100, node.value))
        
        new_mse = calculate_mse(individual, x_data, y_data)
        if new_mse < current_mse and (current_mse - new_mse) < 1e-8:
            break
    
    return individual, calculate_mse(individual, x_data, y_data)


def optimize_constants(individual, x_data, y_data, max_iterations=50):
    constant_nodes = individual.collect_constants()
    if not constant_nodes:
        return individual, calculate_mse(individual, x_data, y_data)
    
    original_values = [node.value for node in constant_nodes]
    
    if HAS_SCIPY:
        def objective_function(params):
            for node, param in zip(constant_nodes, params):
                node.value = param
            mse = calculate_mse(individual, x_data, y_data)
            return mse if mse != float('inf') else 1e10
        
        try:
            bounds = [(-100, 100) for _ in original_values]
            result = minimize(
                objective_function,
                original_values,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iterations, 'ftol': 1e-8}
            )
            
            for node, optimized_value in zip(constant_nodes, result.x):
                node.value = float(optimized_value)
            
            final_mse = calculate_mse(individual, x_data, y_data)
            return individual, final_mse
        except Exception:
            pass
    
    try:
        for node, original_value in zip(constant_nodes, original_values):
            node.value = original_value
        return gradient_descent_optimize(individual, x_data, y_data, constant_nodes, max_iterations)
    except Exception:
        for node, original_value in zip(constant_nodes, original_values):
            node.value = original_value
        return individual, calculate_mse(individual, x_data, y_data)


class GeneticProgramming:
    def __init__(self, pop_size=100, max_depth=6, generations=50, 
                 crossover_rate=0.8, mutation_rate=0.15, elitism=0.1,
                 optimize_constants=True, opt_interval=5, opt_iterations=50,
                 use_dimensional_check=True, target_dimension=None, 
                 var_dimensions=None):
        self.pop_size = pop_size
        self.max_depth = max_depth
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.optimize_constants = optimize_constants
        self.opt_interval = opt_interval
        self.opt_iterations = opt_iterations
        self.use_dimensional_check = use_dimensional_check
        self.target_dimension = target_dimension
        self.var_dimensions = var_dimensions if var_dimensions is not None else {}

    def initialize_population(self):
        population = []
        for _ in range(self.pop_size):
            if random.random() < 0.5:
                individual = generate_full(random.randint(1, self.max_depth))
            else:
                individual = generate_grow(random.randint(1, self.max_depth))
            population.append(individual)
        return population

    def evolve(self, x_data, y_data, verbose=True):
        population = self.initialize_population()
        best_individual = None
        best_fitness = float('inf')
        
        for gen in range(self.generations):
            fitnesses = [
                calculate_fitness(
                    ind, x_data, y_data, 
                    self.target_dimension, self.var_dimensions,
                    self.use_dimensional_check
                ) 
                for ind in population
            ]
            
            current_best_idx = fitnesses.index(min(fitnesses))
            current_best_fitness = fitnesses[current_best_idx]
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx].clone()
            
            if self.optimize_constants and gen % self.opt_interval == 0 and best_individual is not None:
                best_individual, opt_mse = optimize_constants(
                    best_individual, x_data, y_data, 
                    max_iterations=self.opt_iterations
                )
                best_fitness = opt_mse + best_individual.size() * 0.001
            
            if verbose and gen % 5 == 0:
                simplified = simplify_expression(best_individual.clone())
                print(f'Generation {gen}: Best Fitness = {best_fitness:.6f}, Expression = {simplified}')
            
            if best_fitness < 1e-6:
                if verbose:
                    print(f'Converged at generation {gen}!')
                break
            
            new_population = []
            
            elite_count = int(self.pop_size * self.elitism)
            sorted_pop = [ind for _, ind in sorted(zip(fitnesses, population), key=lambda x: x[0])]
            new_population.extend([ind.clone() for ind in sorted_pop[:elite_count]])
            
            while len(new_population) < self.pop_size:
                parent1 = tournament_selection(population, fitnesses)
                parent2 = tournament_selection(population, fitnesses)
                
                if random.random() < self.crossover_rate:
                    child1, child2 = crossover(parent1, parent2)
                    new_population.append(child1)
                    if len(new_population) < self.pop_size:
                        new_population.append(child2)
                else:
                    new_population.append(parent1.clone())
                    if len(new_population) < self.pop_size:
                        new_population.append(parent2.clone())
            
            for i in range(elite_count, len(new_population)):
                if random.random() < self.mutation_rate:
                    new_population[i] = mutate(new_population[i], self.max_depth)
            
            population = new_population
        
        if self.optimize_constants and best_individual is not None:
            best_individual, _ = optimize_constants(
                best_individual, x_data, y_data,
                max_iterations=self.opt_iterations * 2
            )
        
        best_individual = simplify_expression(best_individual)
        final_mse = calculate_mse(best_individual, x_data, y_data)
        
        return best_individual, final_mse


def symbolic_regression(x_data, y_data, pop_size=200, max_depth=6, generations=100, 
                        crossover_rate=0.85, mutation_rate=0.15, elitism=0.1, 
                        optimize_constants=True, opt_interval=5, opt_iterations=50,
                        use_dimensional_check=True, target_dimension=None,
                        var_dimensions=None, verbose=True, **kwargs):
    gp = GeneticProgramming(pop_size=pop_size, max_depth=max_depth, 
                           generations=generations, crossover_rate=crossover_rate,
                           mutation_rate=mutation_rate, elitism=elitism,
                           optimize_constants=optimize_constants,
                           opt_interval=opt_interval,
                           opt_iterations=opt_iterations,
                           use_dimensional_check=use_dimensional_check,
                           target_dimension=target_dimension,
                           var_dimensions=var_dimensions)
    best_expr, best_fitness = gp.evolve(x_data, y_data, verbose=verbose)
    return best_expr, best_fitness


def predict(expr, x):
    return expr.evaluate(x)


def print_predictions(expr, x_data, y_data):
    print('\nPredictions:')
    print(f'{"x":>8} {"Actual":>10} {"Predicted":>12} {"Error":>10}')
    print('-' * 45)
    total_error = 0
    for x, y_target in zip(x_data, y_data):
        y_pred = expr.evaluate(x)
        error = abs(y_pred - y_target)
        total_error += error
        print(f'{x:>8.2f} {y_target:>10.4f} {y_pred:>12.4f} {error:>10.4f}')
    print('-' * 45)
    print(f'Mean Absolute Error: {total_error / len(x_data):.6f}')


if __name__ == '__main__':
    print('=' * 60)
    print('Example 1: Symbolic Regression for f(x) = x^2')
    print('=' * 60)
    x_data = [i for i in range(-5, 6)]
    y_data = [x**2 for x in x_data]
    print(f'Training data points: {len(x_data)}')
    print(f'x = {x_data}')
    print(f'y = {y_data}')
    print()
    
    expr, fitness = symbolic_regression(x_data, y_data, pop_size=150, generations=80, max_depth=5)
    
    print(f'\nFinal Best Expression: {expr}')
    print(f'Final MSE Fitness: {fitness:.6f}')
    print_predictions(expr, x_data, y_data)
    
    print('\n\n' + '=' * 60)
    print('Example 2: Symbolic Regression for f(x) = sin(x) + 2*x')
    print('=' * 60)
    x_data2 = [i * 0.4 for i in range(-8, 9)]
    y_data2 = [math.sin(x) + 2 * x for x in x_data2]
    print(f'Training data points: {len(x_data2)}')
    print()
    
    expr2, fitness2 = symbolic_regression(x_data2, y_data2, pop_size=200, generations=120, max_depth=6)
    
    print(f'\nFinal Best Expression: {expr2}')
    print(f'Final MSE Fitness: {fitness2:.6f}')
    print_predictions(expr2, x_data2, y_data2)
    
    print('\n\n' + '=' * 60)
    print('Example 3: Demonstrating Constant Optimization Effect')
    print('=' * 60)
    print('Target function: f(x) = 3.14 * x + 2.718')
    x_data3 = [i * 0.5 for i in range(-6, 7)]
    y_data3 = [3.14 * x + 2.718 for x in x_data3]
    print(f'Training data points: {len(x_data3)}')
    print()
    
    print('Running WITH constant optimization...')
    expr_opt, fitness_opt = symbolic_regression(
        x_data3, y_data3, 
        pop_size=150, generations=80, max_depth=4,
        optimize_constants=True, opt_interval=4,
        verbose=False
    )
    print(f'Expression (with opt): {expr_opt}')
    print(f'MSE (with opt): {fitness_opt:.8f}')
    
    print('\nRunning WITHOUT constant optimization...')
    expr_no_opt, fitness_no_opt = symbolic_regression(
        x_data3, y_data3, 
        pop_size=150, generations=80, max_depth=4,
        optimize_constants=False,
        verbose=False
    )
    print(f'Expression (no opt): {expr_no_opt}')
    print(f'MSE (no opt): {fitness_no_opt:.8f}')
    
    print(f'\nImprovement: {(fitness_no_opt - fitness_opt) / fitness_no_opt * 100:.2f}%')
    
    print('\n\n' + '=' * 60)
    print('Example 4: Dimensional Consistency Check Demo')
    print('=' * 60)
    
    print('\nDemonstrating dimensional validation:')
    print('-' * 40)
    
    t = Dimension(time=1)
    L = Dimension(length=1)
    v = Dimension(length=1, time=-1)
    a = Dimension(length=1, time=-2)
    
    print(f"Time (t): {t}")
    print(f"Length (x): {L}")
    print(f"Velocity (v): {v}")
    print(f"Acceleration (a): {a}")
    
    expr1 = Node('+', Node('x'), Node(5.0))
    is_ok, dim = check_dimensional_correctness(expr1, var_dimensions={'x': L})
    print(f'\nExpression: x + 5')
    print(f'Dimensionally consistent: {is_ok}, Result dimension: {dim}')
    
    expr2 = Node('*', Node('v'), Node('t'))
    is_ok, dim = check_dimensional_correctness(
        expr2, var_dimensions={'v': v, 't': t}, target_dimension=L
    )
    print(f'\nExpression: v * t')
    print(f'Dimensionally consistent: {is_ok}, Result dimension: {dim}')
    print(f'Result is length: {dim == L}')
    
    expr3 = Node('+', Node('x'), Node('t'))
    is_ok, dim = check_dimensional_correctness(
        expr3, var_dimensions={'x': L, 't': t}
    )
    print(f'\nExpression: x + t (length + time - INVALID)')
    print(f'Dimensionally consistent: {is_ok}')
    
    expr4 = Node('sin', Node('x'))
    is_ok, dim = check_dimensional_correctness(
        expr4, var_dimensions={'x': L}
    )
    print(f'\nExpression: sin(x) (sin of length - INVALID)')
    print(f'Dimensionally consistent: {is_ok}')
    
    print('\n\n' + '=' * 60)
    print('Usage Example with Dimensional Check')
    print('=' * 60)
    print('''
from genetic_programming import (
    symbolic_regression, predict, optimize_constants,
    Dimension, check_dimensional_correctness
)

# Define dimensions
length_dim = Dimension(length=1)
time_dim = Dimension(time=1)
velocity_dim = Dimension(length=1, time=-1)

# Variable dimension mapping
var_dims = {
    'x': length_dim,
    't': time_dim
}

# Target dimension for s = v * t
target_dim = length_dim

# Your data points (e.g., distance vs time)
x_data = [1, 2, 3, 4, 5]
y_data = [2.5 * t for t in x_data]  # s = 2.5 * t

# Run genetic programming with dimensional checks
expr, fitness = symbolic_regression(
    x_data, y_data,
    pop_size=150,
    generations=80,
    optimize_constants=True,
    use_dimensional_check=True,      # Enable dimensional checks
    target_dimension=target_dim,      # Expected result dimension
    var_dimensions=var_dims,          # Variable dimension definitions
    verbose=True
)

print(f'\\nFound expression: {expr}')
print(f'Fitness (MSE + penalties): {fitness:.6f}')

# Verify dimensional correctness of the result
is_consistent, result_dim = check_dimensional_correctness(
    expr, target_dim, var_dims
)
print(f'Dimensionally consistent: {is_consistent}')
print(f'Result dimension: {result_dim}')
''')
