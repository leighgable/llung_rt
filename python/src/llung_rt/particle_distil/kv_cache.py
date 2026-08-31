import torch

class DynamicSMCache:
    def __init__(
        self,
        num_particles: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_length: int,
        device: str = "cuda",
    ):
        self.num_particles = num_particles
        self.num_layers = num_layers
        self.max_length = max_length

        self.k_cache = torch.zeros(
            (num_layers, num_particles, num_heads, max_length, head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        self.v_cache = torch.zeros(
            (num_layers, num_particles, num_heads, max_length, head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        self.scratch_k = torch.empty_like(self.k_cache)
        self.scratch_v = torch.empty_like(self.v_cache)

    def resample_in_place(
        self,
        ancestor_indices: torch.Tensor,
        current_seq_length: int, 
    ):
        """ resample particles along batch dimension """
        
        k_active = self.k_cache[:, :, :, :current_seq_length, :]
        v_active = self.v_cache[:, :, :, :current_seq_length, :]
        
        torch.index_select(k_active,
                           dim=1,
                           index=ancestor_indices,
                           out=self.scratch_k,
                        )
        torch.index_select(v_active,
                           dim=1,
                           index=ancestor_indices,
                           out=self.scratch_v,
                        )
        k_active.copy_(self.scratch_k[:, :, :, :current_seq_length, :])
        v_active.copy_(self.scratch_k[:, :, :, :current_seq_length, :])

    class PagedSMCache:
        def __init__(
            self,
            num_particles: int,
            num_layers: int,
            max_length: int,
            block_size: int = 16,
            device: str = "cuda",
        ):
            self.num_particles = num_particles
            self.num_blocks_per_particle = (max_length + block_size - 1) // block_size

            self.block_table = torch.arange(
                num_particles * self.num_blocks_per_particle, device=device,
            ).reshape(num_particles, self.num_blocks_per_particle)

        def resample_pointers_only(
            self,
            ancestor_indices: torch.Tensor,
            ):
            """ resampling by re-indexing pointers """
            self.block_table = self.block_table[ancestor_indices]

    
