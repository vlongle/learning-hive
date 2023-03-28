
num_agents = 4

Monolithic MNIST:
 AFTER LOCAL
{'components.0.weight': 0.021810632199048996, 'components.0.bias': 0.013573450967669487, 'components.1.weight': 0.01388462446630001, 'components.1.bias': 0.007945799268782139, 'random_linear_projection.weight': 0.0, 'random_linear_projection.bias': 0.0}
Is it possible to sync these models after such divergence?

Modular MNIST: the models add a new components but previous components are basically fixed.

{'components.0.weight': 0.0024712281301617622, 'components.0.bias': 0.0019073376897722483, 'components.1.weight': 0.002378673292696476, 'components.1.bias': 0.0019012467237189412, 'components.2.weight': 0.1001572534441948, 'components.2.bias': 0.088865265250206, 'random_linear_projection.weight': 0.0, 'random_linear_projection.bias': 0.0}