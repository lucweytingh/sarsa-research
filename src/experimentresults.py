import json
from collections import defaultdict
from pathlib import Path

from dotenv import dotenv_values

from src.sarsa import NAME2ALG
from src.utils import ENV_NAMES
from src.constants import ENV_NAMES
from src.utils import load


DOTENV_KEY2VAl = dotenv_values()
PROJECT_DIR = Path(DOTENV_KEY2VAl["PROJECT_DIR"])


class ExperimentResults(dict):
    ENV2ALG2METADATA_PATH = PROJECT_DIR / "storage/env2alg2metadata.json"

    def __init__(self, env2alg2metadata):
        self.update(env2alg2metadata)

    def _get_env2alg2feature(self, feature):
        env2alg2feature = defaultdict(dict)
        for env, alg2metadata in self.items():
            for alg, metadata in alg2metadata.items():
                env2alg2feature[env][alg] = metadata.get(feature, None)
        return dict(env2alg2feature)

    def _get_feature(self, feature, env, alg):
        return self[env][alg].get(feature, None)

    @property
    def env2alg2nof_alphas(self):
        return self._get_env2alg2feature("nof_alphas")

    @property
    def env2alg2seeds(self):
        return self._get_env2alg2feature("seeds")

    @property
    def env2alg2opt_alpha(self):
        return self._get_env2alg2feature("opt_alpha")

    def write(self):
        with open(self.ENV2ALG2METADATA_PATH, "w") as f:
            f.write(json.dumps(self))

    def results_present(self, env_name, nof_alphas, seeds):
        "checks if results are present for these parameters"
        results_present = True
        for alg_name in NAME2ALG.keys():
            results_present &= (
                self._get_feature("nof_alphas", env_name, alg_name)
                == nof_alphas
            )
            results_present &= self._get_feature(
                "seeds", env_name, alg_name
            ) == sorted(seeds)
        return results_present

    def show_results(self, env_name):
        alg2metadata = self[env_name]
        for alg, metadata in alg2metadata.items():
            print(
                f"Optimal alpha for {alg} computed with seeds={metadata['seeds']} and nof_alphas={metadata['nof_alphas']}\t{metadata['opt_alpha']}"
            )

    @classmethod
    def from_storage(cls):
        env2alg2meta_data = {
            env_name: {algname: dict() for algname in NAME2ALG.keys()}
            for env_name in ENV_NAMES
        }
        if cls.ENV2ALG2METADATA_PATH.exists():
            data = {**env2alg2meta_data, **load(cls.ENV2ALG2METADATA_PATH)}
        else:
            data = env2alg2meta_data
        return cls(data)
