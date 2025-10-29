# price_optimizer.py
import numpy as np

class ParticleSwarmOptimizer:
    def __init__(self, objective_function, bounds, num_particles, max_iter, verbose=True):
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.verbose = verbose
        self.num_dimensions = len(self.bounds)
        
        # Initialize swarm
        self.particles_pos = np.random.rand(self.num_particles, self.num_dimensions) * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0]
        self.particles_vel = np.random.randn(self.num_particles, self.num_dimensions) * 0.1
        
        self.pbest_pos = self.particles_pos.copy()
        self.pbest_val = np.array([float('inf')] * self.num_particles)
        
        self.gbest_pos = None
        self.gbest_val = float('inf')

    def optimize(self):
        for i in range(self.max_iter):
            for j in range(self.num_particles):
                fitness = self.objective_function(self.particles_pos[j])
                
                if fitness < self.pbest_val[j]:
                    self.pbest_pos[j] = self.particles_pos[j].copy()
                    self.pbest_val[j] = fitness
                
                if fitness < self.gbest_val:
                    self.gbest_pos = self.particles_pos[j].copy()
                    self.gbest_val = fitness

            # Update velocities and positions
            w, c1, c2 = 0.5, 1.5, 1.5
            
            for j in range(self.num_particles):
                r1, r2 = np.random.rand(self.num_dimensions), np.random.rand(self.num_dimensions)
                cognitive_vel = c1 * r1 * (self.pbest_pos[j] - self.particles_pos[j])
                social_vel = c2 * r2 * (self.gbest_pos - self.particles_pos[j])
                
                self.particles_vel[j] = w * self.particles_vel[j] + cognitive_vel + social_vel
                self.particles_pos[j] += self.particles_vel[j]
                
                # Handle boundaries
                self.particles_pos[j] = np.maximum(self.particles_pos[j], self.bounds[:, 0])
                self.particles_pos[j] = np.minimum(self.particles_pos[j], self.bounds[:, 1])

            if (i + 1) % 10 == 0:
                if self.verbose:
                    print(f"Iteration {i+1}/{self.max_iter}, Best Profit Found: {-self.gbest_val:.2f}")

        print("\nOptimization finished.")
        return self.gbest_pos, -self.gbest_val