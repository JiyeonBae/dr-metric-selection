import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from distutils.spawn import find_executable
from glob import glob
import psutil

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class OctaveProjection(BaseEstimator, TransformerMixin):
    def __init__(self, projection, command, verbose):
        self.known_projections = [
            'ProbPCA', 'GDA', 'LLC', 'ManifoldChart', 'CFA',
            'MVU', 'FastMVU', 'CCA', 'LandmarkMVU', 'GPLVM',
            'NCA', 'MCML', 'LMNN', 'Sammon']
        self.projection = projection
        self.command = command
        self.verbose = verbose

        if self.projection not in self.known_projections:
            raise ValueError('Invalid projection name: %s. Valid values are %s' % 
                             (self.projection, ','.join(self.known_projections)))

    def fit_transform(self, X, y=None):
        raise Exception('Not implemented')

    def _send_data(self, X, y):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.tmp_file = tempfile.NamedTemporaryFile(
            mode='w', dir=self.tmp_dir.name, suffix='.data', delete=False)
        
        try:
            np.savetxt(self.tmp_file.name, X, delimiter=',')
        finally:
            self.tmp_file.close()
        return self.tmp_dir.name

    def _receive_data(self):
        proj_file = self.tmp_file.name + '.prj'
        if not os.path.exists(proj_file):
            raise ValueError('Error looking for projection file %s' % proj_file)
        X_new = np.loadtxt(proj_file, delimiter=',')
        self.safe_delete(self.tmp_file.name)
        return X_new

    def release_file(self, file_path):
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for open_file in proc.open_files():
                    if open_file.path == file_path:
                        proc.terminate()
                        proc.wait()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, FileNotFoundError):
                continue

    def safe_delete(self, file_path, retries=5, delay=1):
        for attempt in range(retries):
            try:
                self.release_file(file_path)
                os.unlink(file_path)
                return
            except PermissionError:
                print(f"Attempt {attempt + 1} failed to delete {file_path}. Retrying...")
        print(f"Failed to delete {file_path} after {retries} attempts.")

    def _run(self, X, y, cmdargs):
        if self.verbose:
            print("Sending data to Octave...")
        self._send_data(X, y)

        libpath = os.path.dirname(self.command)
        # Set the path to your Octave installation directory
        # Change "C:/Octave/Octave-4.4.1/bin/octave.bat" to the correct Octave installation path
        octave_path = "C:/Octave/Octave-4.4.1/bin/octave.bat"
        cmdline = [
            octave_path, '--built-in-docstrings-file', 'built-in-docstrings', '-qf',
            self.command, libpath, self.tmp_file.name, self.projection
        ] + [str(x) for x in cmdargs]

        if self.verbose:
            print('Running Octave command...')

        try:
            rc = subprocess.run(
                cmdline, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=86400, check=True
            )
            if self.verbose:
                print("Octave command finished successfully.")
                if rc.stdout.strip():
                    print('Octave stdout:', rc.stdout)
                if rc.stderr.strip():
                    print('Octave stderr:', rc.stderr)
        except subprocess.CalledProcessError as e:
            print("Error: Octave execution failed.")
            if self.verbose:
                print(f"Return code: {e.returncode}")
                print(f"Output: {e.output}")
                print(f"Error output: {e.stderr}")
            raise

        try:
            if self.verbose:
                print("Receiving projected data...")
            X_new = self._receive_data()
            if self.verbose:
                print("Projection data received successfully.")
            return X_new
        except Exception:
            print("Error: Failed during data retrieval.")
            if self.verbose:
                print(traceback.format_exc())
            raise Exception('Error running projection')


class ProbPCA(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', max_iter=200, verbose=False):
        super(ProbPCA, self).__init__(
            projection='ProbPCA', command=command, verbose=verbose)
        self.set_params(command, max_iter, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', max_iter=200, verbose=False):
        self.command = command
        self.verbose = verbose
        self.max_iter = max_iter

    def fit_transform(self, X, y=None):
        return super(ProbPCA, self)._run(X, y, [self.max_iter])


class GDA(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', kernel='gauss', verbose=False):
        super(GDA, self).__init__(
            projection='GDA', command=command, verbose=verbose)
        self.set_params(command, kernel, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', kernel='gauss', verbose=False):
        self.command = command
        self.verbose = verbose
        self.kernel = kernel

    def fit_transform(self, X, y=None):
        return super(GDA, self)._run(X, y, [self.kernel])


class MCML(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', verbose=False):
        super(MCML, self).__init__(
            projection='MCML', command=command, verbose=verbose)
        self.set_params(command, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', verbose=False):
        self.command = command
        self.verbose = verbose

    def fit_transform(self, X, y=None):
        return super(MCML, self)._run(X, y, [])


class Sammon(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', verbose=False):
        super(Sammon, self).__init__(
            projection='Sammon', command=command, verbose=verbose)
        self.set_params(command, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', verbose=False):
        self.command = command
        self.verbose = verbose

    def fit_transform(self, X, y=None):
        return super(Sammon, self)._run(X, y, [])


class LMNN(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', k=3, verbose=False):
        super(LMNN, self).__init__(
            projection='LMNN', command=command, verbose=verbose)
        self.set_params(command, k, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', k=3, verbose=False):
        self.command = command
        self.verbose = verbose
        self.k = k

    def fit_transform(self, X, y=None):
        return super(LMNN, self)._run(X, y, [self.k])


class MVU(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', k=12, verbose=False):
        super(MVU, self).__init__(
            projection='MVU', command=command, verbose=verbose)
        self.set_params(command, k, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', k=12, verbose=False):
        self.command = command
        self.verbose = verbose
        self.k = k

    def fit_transform(self, X, y=None):
        return super(MVU, self)._run(X, y, [self.k])


class FastMVU(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', k=12, verbose=False):
        super(FastMVU, self).__init__(
            projection='FastMVU', command=command, verbose=verbose)
        self.set_params(command, k, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', k=12, verbose=False):
        self.command = command
        self.verbose = verbose
        self.k = k

    def fit_transform(self, X, y=None):
        return super(FastMVU, self)._run(X, y, [self.k])


class LandmarkMVU(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', k1=3, k2=12, verbose=False):
        super(LandmarkMVU, self).__init__(
            projection='LandmarkMVU', command=command, verbose=verbose)
        self.set_params(command, k1, k2, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', k1=3, k2=12, verbose=False):
        self.command = command
        self.verbose = verbose
        self.k1 = k1
        self.k2 = k2

    def fit_transform(self, X, y=None):
        return super(LandmarkMVU, self)._run(X, y, [self.k1, self.k2])


class CCA(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', k=12, verbose=False):
        super(CCA, self).__init__(
            projection='CCA', command=command, verbose=verbose)
        self.set_params(command, k, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', k=12, verbose=False):
        self.command = command
        self.verbose = verbose
        self.k = k

    def fit_transform(self, X, y=None):
        return super(CCA, self)._run(X, y, [self.k])


class GPLVM(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', sigma=1.0, verbose=False):
        super(GPLVM, self).__init__(
            projection='GPLVM', command=command, verbose=verbose)
        self.set_params(command, sigma, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', sigma=1.0, verbose=False):
        self.command = command
        self.verbose = verbose
        self.sigma = sigma

    def fit_transform(self, X, y=None):
        return super(GPLVM, self)._run(X, y, [self.sigma])


class NCA(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', lambd=0.0, verbose=False):
        super(NCA, self).__init__(
            projection='NCA', command=command, verbose=verbose)
        self.set_params(command, lambd, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', lambd=0.0, verbose=False):
        self.command = command
        self.verbose = verbose
        self.lambd = lambd

    def fit_transform(self, X, y=None):
        return super(NCA, self)._run(X, y, [self.lambd])


class LLC(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', k=12, n_analyzers=20, max_iter=200, verbose=False):
        super(LLC, self).__init__(
            projection='LLC', command=command, verbose=verbose)
        self.set_params(command, k, n_analyzers, max_iter, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', k=12, n_analyzers=20, max_iter=200, verbose=False):
        self.command = command
        self.verbose = verbose
        self.k = k
        self.n_analyzers = n_analyzers
        self.max_iter = max_iter

    def fit_transform(self, X, y=None):
        return super(LLC, self)._run(X, y, [self.k, self.n_analyzers, self.max_iter])


class ManifoldChart(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', n_analyzers=40, max_iter=200, verbose=False):
        super(ManifoldChart, self).__init__(
            projection='ManifoldChart', command=command, verbose=verbose)
        self.set_params(command, n_analyzers, max_iter, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', n_analyzers=40, max_iter=200, verbose=False):
        self.command = command
        self.verbose = verbose
        self.n_analyzers = n_analyzers
        self.max_iter = max_iter

    def fit_transform(self, X, y=None):
        return super(ManifoldChart, self)._run(X, y, [self.n_analyzers, self.max_iter])


class CFA(OctaveProjection):
    def __init__(self, command=os.getcwd() + '/drtoolbox/drrun.m', n_analyzers=40, max_iter=200, verbose=False):
        super(CFA, self).__init__(
            projection='CFA', command=command, verbose=verbose)
        self.set_params(command, n_analyzers, max_iter, verbose)

    def set_params(self, command=os.getcwd() + '/drtoolbox/drrun.m', n_analyzers=40, max_iter=200, verbose=False):
        self.command = command
        self.verbose = verbose
        self.n_analyzers = n_analyzers
        self.max_iter = max_iter

    def fit_transform(self, X, y=None):
        return super(CFA, self)._run(X, y, [self.n_analyzers, self.max_iter])