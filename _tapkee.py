import os
import numpy as np
from glob import glob
import tempfile
import subprocess
import shutil
import psutil
import time
from distutils.spawn import find_executable
from sklearn.base import BaseEstimator, TransformerMixin

class TapkeeCLIProjection(BaseEstimator, TransformerMixin):
    def __init__(self, projection, command="./tapkee/tapkee", verbose=False):
        self.known_projections = [
            'diffusion_map',
            'manifold_sculpting',
            'stochastic_proximity_embedding',
            'locality_preserving_projections',
            'linear_local_tangent_space_alignment',
            'neighborhood_preserving_embedding',
            'landmark_multidimensional_scaling'
        ]

        self.projection = projection
        self.verbose = verbose

        self.command = self._to_wsl_path(os.path.abspath(command))

        if self.projection not in self.known_projections:
            raise ValueError(f"Invalid projection name: {self.projection}. "
                             f"Valid values are {', '.join(self.known_projections)}")

        if verbose:
            print(f"Debug: Initialized with command: {self.command}")

    def _to_wsl_path(self, path):
        if ":" in path:
            drive, rest = path.split(":", 1)
            rest = rest.replace("\\", "/")
            path = f"/mnt/{drive.lower()}{rest}"
        return path

    def fit_transform(self, X, y=None):
        raise Exception("Not implemented")

    def _send_data(self, X, y):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_file = tempfile.NamedTemporaryFile(
            mode='w', dir=self.tmp_dir.name, suffix='.data', delete=False
        )
        try:
            np.savetxt(self.tmp_file.name, X, delimiter=',')
            if self.verbose:
                print(f"Data saved to: {self.tmp_file.name}")
                print(f"Temporary directory: {self.tmp_dir.name}")
        except Exception as e:
            print(f"Error saving data to temporary file: {e}")
            raise
        finally:
            self.tmp_file.close()
        return self.tmp_dir.name

    def _receive_data(self):
        proj_file = self.tmp_file.name + '.prj'

        if self.verbose:
            print(f"Debug: Looking for projection file at {proj_file}")

        if not os.path.exists(proj_file):
            raise ValueError(f"Projection file not found: {proj_file}")

        try:
            X_new = np.loadtxt(proj_file, delimiter=',')
            if self.verbose:
                print(f"Debug: Projection file loaded successfully with shape {X_new.shape}")
            return X_new
        except Exception as e:
            print(f"Error loading projection file: {e}")
            raise
        finally:
            self.safe_delete(self.tmp_file.name)
            self.safe_delete(proj_file)

    def safe_delete(self, file_path, retries=5, delay=1):
        for attempt in range(retries):
            try:
                os.unlink(file_path)
                if self.verbose:
                    print(f"Debug: Successfully deleted {file_path}")
                return
            except PermissionError:
                print(f"Attempt {attempt + 1} failed to delete {file_path}. Retrying...")
                time.sleep(delay)
        print(f"Failed to delete {file_path} after {retries} attempts.")

    def _run(self, X, y, cmdargs):
        tmp_dir = self._send_data(X, y)
        tmp_file_path = os.path.join(tmp_dir, os.path.basename(self.tmp_file.name))

        input_path_wsl = tmp_file_path.replace("\\", "/").replace("C:/", "/mnt/c/")
        output_path_wsl = input_path_wsl + ".prj"

        safe_cmdargs = [str(arg) if isinstance(arg, (int, float)) else arg for arg in cmdargs]

        wsl_command = [
            "wsl", "--", self.command,
            "--method", self.projection,
            "--eigen-method", "dense",
            "--neighbors-method", "brute",
            "-i", input_path_wsl,
            "-o", output_path_wsl
        ] + safe_cmdargs

        if self.verbose:
            print(f"Debug: WSL command prepared: {' '.join(wsl_command)}")

        try:
            rc = subprocess.run(
                wsl_command,
                universal_newlines=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=86400,
                check=True
            )
            if self.verbose:
                print("Debug: Command executed successfully")
        except subprocess.CalledProcessError as e:
            print("Error occurred while running tapkee in WSL:")
            print("stdout:", e.stdout)
            print("stderr:", e.stderr)
            raise e

        try:
            return self._receive_data()
        except Exception as e:
            print(f"Error during data retrieval: {e}")
            raise

class DiffusionMaps(TapkeeCLIProjection):
    def __init__(self, command="./tapkee/tapkee", t=2, width=1.0, verbose=False):
        super(DiffusionMaps, self).__init__(
            projection="diffusion_map", command=command, verbose=verbose)
        self.set_params(command, t, width, verbose)

    def set_params(self, command="./tapkee/tapkee", t=2, width=1.0, verbose=False):
        self.command = command
        self.verbose = verbose
        self.t = t
        self.width = width

    def fit_transform(self, X, y=None):
        return super(DiffusionMaps, self)._run(X, y,
                                               ["--timesteps", str(self.t),
                                                "--gaussian-width", str(self.width)])

class StochasticProximityEmbedding(TapkeeCLIProjection):
    def __init__(self, command="./tapkee/tapkee", n_neighbors=12, n_updates=20, max_iter=0, verbose=False):
        super(StochasticProximityEmbedding, self).__init__(
            projection='stochastic_proximity_embedding', command=command, verbose=verbose)
        self.set_params(command, n_neighbors, n_updates, max_iter, verbose)

    def set_params(self, command="./tapkee/tapkee", n_neighbors=12, n_updates=20, max_iter=0, verbose=False):
        self.command = command
        self.verbose = verbose
        self.n_neighbors = n_neighbors
        self.n_updates = n_updates
        self.max_iter = max_iter

    def fit_transform(self, X, y=None):
        return super(StochasticProximityEmbedding, self)._run(X, y,
                                                              ['-k', self.n_neighbors,
                                                               '--spe-num-updates', self.n_updates,
                                                               '--max-iters', self.max_iter])

class LocalityPreservingProjections(TapkeeCLIProjection):
    def __init__(self, command="./tapkee/tapkee", n_neighbors=10, verbose=False):
        super(LocalityPreservingProjections, self).__init__(
            projection='locality_preserving_projections', command=command, verbose=verbose)
        self.set_params(n_neighbors, verbose)

    def set_params(self, n_neighbors=10, verbose=False):
        self.n_neighbors = n_neighbors
        self.verbose = verbose

    def fit_transform(self, X, y=None):
        return super(LocalityPreservingProjections, self)._run(X, y, 
                                                               ['-k', self.n_neighbors])


class LinearLocalTangentSpaceAlignment(TapkeeCLIProjection):
    def __init__(self, command="./tapkee/tapkee", n_neighbors=10, verbose=False):
        super(LinearLocalTangentSpaceAlignment, self).__init__(
            projection='linear_local_tangent_space_alignment', command=command, verbose=verbose)
        self.set_params(command, n_neighbors, verbose)

    def set_params(self, command="./tapkee/tapkee", n_neighbors=10, verbose=False):
        self.command = command
        self.verbose = verbose
        self.n_neighbors = n_neighbors

    def fit_transform(self, X, y=None):
        return super(LinearLocalTangentSpaceAlignment, self)._run(X, y, 
                                                                  ['-k', self.n_neighbors])
    
class NeighborhoodPreservingEmbedding(TapkeeCLIProjection):
    def __init__(self, command="./tapkee/tapkee", n_neighbors=10, verbose=False):
        super(NeighborhoodPreservingEmbedding, self).__init__(
            projection='neighborhood_preserving_embedding', command=command, verbose=verbose)
        self.set_params(command, n_neighbors, verbose)

    def set_params(self, command="./tapkee/tapkee", n_neighbors=10, verbose=False):
        self.command = command
        self.verbose = verbose
        self.n_neighbors = n_neighbors

    def fit_transform(self, X, y=None):
        return super(NeighborhoodPreservingEmbedding, self)._run(X, y, 
                                                                 ['-k', self.n_neighbors])

class LandmarkMDS(TapkeeCLIProjection):
    def __init__(self, command="./tapkee/tapkee", n_neighbors=10, verbose=False):
        super(LandmarkMDS, self).__init__(
            projection='landmark_multidimensional_scaling', command=command, verbose=verbose)
        self.set_params(command, n_neighbors, verbose)

    def set_params(self, command="./tapkee/tapkee", n_neighbors=10, verbose=False):
        self.command = command
        self.verbose = verbose
        self.n_neighbors = n_neighbors

    def fit_transform(self, X, y=None):
        return super(LandmarkMDS, self)._run(X, y, 
                                             ['-k', self.n_neighbors])
