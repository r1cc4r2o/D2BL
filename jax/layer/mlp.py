import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import logsumexp
import time


class MLP():
    """Simple MLP which has dynamic weights of
        the representations extracted in the 
        previous layers.
    

    
    """
    def __init__(
            self,
            layer_sizes = [784, 512, 512, 10],
            lr: float = 0.01,
            epochs: int = 10,
            batch_size: int = 128,
            n_class: int = 10
            
            ):
        super(MLP, self).__init__()

        self.params = self.init_network_params(layer_sizes, random.PRNGKey(0))
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_class = n_class

        # Batched version of the `predict` function
        self.batched_predict = vmap(self.predict, in_axes=(None, 0))


    def random_layer_params(self, m, n, key, scale=1e-2):
        """Initialize the weights and the biases

                Input:
                    m:
                    n:
                    key:
                    scale:

                Output:
                    weights:
                    biases:
        
        """
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
    
    def init_network_params(self, sizes, key):
        """Initialize the weights and the biases
            of the fully-connected network.

                Input:
                    size:
                    key:

                Output:
                    weights, biases: 
        
        """
        keys = random.split(key, len(sizes))
        return [self.random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
    
    def relu(self, x):
        return jnp.maximum(0, x)
    
    def predict(self, params, image):
        # per-example predictions
        activations = image
        for w, b in params[:-1]:
            outputs = jnp.dot(w, activations) + b
            activations = self.relu(outputs)
        
        final_w, final_b = params[-1]
        logits = jnp.dot(final_w, activations) + final_b
        return logits - logsumexp(logits)
    
    def one_hot(self, x, k, dtype=jnp.float32):
        """Create a one-hot encoding of x of size k."""
        return jnp.array(x[:, None] == jnp.arange(k), dtype)
        
    def accuracy(self, params, images, targets):
        target_class = jnp.argmax(targets, axis=1)
        predicted_class = jnp.argmax(self.batched_predict(params, images), axis=1)
        return jnp.mean(predicted_class == target_class)

    def loss(self, params, images, targets):
        preds = self.batched_predict(params, images)
        return -jnp.mean(preds * targets)

    @jit
    def update(self, params, x, y):
        grads = grad(self.loss)(params, x, y)
        return [(w - self.lr * dw, b - self.lr * db)
                    for (w, b), (dw, db) in zip(self.params, grads)]
    
    def train(self, )
        for epoch in range(self.epochs):
            start_time = time.time()
            for x, y in self.get_train_batches():
                x = jnp.reshape(x, (len(x), num_pixels))
                y = one_hot(y, num_labels)
                params = update(params, x, y)
            epoch_time = time.time() - start_time
    def forward(self, x):
        
        


        return x