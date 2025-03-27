### Issues
* training error:
```
// ...existing code...
################################################################################
                       Learning iteration 67/1000                       

                       Computation: 51585 steps/s (collection: 0.361s, learning 0.115s)
               Value function loss: 0.0010
                    Surrogate loss: -0.0068
             Mean action noise std: 0.66
                 Mean total reward: -1.30
               Mean episode length: 360.00
Episode_Reward/end_effector_position_tracking: -0.0440
Episode_Reward/end_effector_position_tracking_fine_grained: 0.0071
Episode_Reward/end_effector_orientation_tracking: -0.0662
        Episode_Reward/action_rate: -0.0008
          Episode_Reward/joint_vel: -0.0022
    Metrics/ee_pose/position_error: 0.1996
 Metrics/ee_pose/orientation_error: 0.4540
      Episode_Termination/time_out: 2.7917
--------------------------------------------------------------------------------
                   Total timesteps: 1671168
                    Iteration time: 0.48s
                        Total time: 30.82s
                               ETA: 422.8s

Error executing job with overrides: []
Traceback (most recent call last):
  File "/home/check/miniconda3/envs/env_isaaclab/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/home/check/miniconda3/envs/env_isaaclab/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/home/check/.vscode/extensions/ms-python.debugpy-2025.4.1-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher/../../debugpy/__main__.py", line 71, in <module>
    cli.main()
  File "/home/check/.vscode/extensions/ms-python.debugpy-2025.4.1-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher/../../debugpy/../debugpy/server/cli.py", line 501, in main
    run()
  File "/home/check/.vscode/extensions/ms-python.debugpy-2025.4.1-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher/../../debugpy/../debugpy/server/cli.py", line 351, in run_file
    runpy.run_path(target, run_name="__main__")
  File "/home/check/.vscode/extensions/ms-python.debugpy-2025.4.1-linux-x64/bundled/libs/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 310, in run_path
    return _run_module_code(code, init_globals, run_name, pkg_name=pkg_name, script_name=fname)
  File "/home/check/.vscode/extensions/ms-python.debugpy-2025.4.1-linux-x64/bundled/libs/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 127, in _run_module_code
    _run_code(code, mod_globals, init_globals, mod_name, mod_spec, pkg_name, script_name)
  File "/home/check/.vscode/extensions/ms-python.debugpy-2025.4.1-linux-x64/bundled/libs/debugpy/_vendored/pydevd/_pydevd_bundle/pydevd_runpy.py", line 118, in _run_code
    exec(code, run_globals)
  File "scripts/reinforcement_learning/rsl_rl/train.py", line 154, in <module>
    main()
  File "/home/check/Documents/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py", line 104, in wrapper
    hydra_main()
  File "/home/check/miniconda3/envs/env_isaaclab/lib/python3.10/site-packages/hydra/main.py", line 94, in decorated_main
    _run_hydra(
  File "/home/check/miniconda3/envs/env_isaaclab/lib/python3.10/site-packages/hydra/_internal/utils.py", line 394, in _run_hydra
    _run_app(
  File "/home/check/miniconda3/envs/env_isaaclab/lib/python3.10/site-packages/hydra/_internal/utils.py", line 457, in _run_app
    run_and_report(
  File "/home/check/miniconda3/envs/env_isaaclab/lib/python3.10/site-packages/hydra/_internal/utils.py", line 223, in run_and_report
    raise ex
  File "/home/check/miniconda3/envs/env_isaaclab/lib/python3.10/site-packages/hydra/_internal/utils.py", line 220, in run_and_report
    return func()
  File "/home/check/miniconda3/envs/env_isaaclab/lib/python3.10/site-packages/hydra/_internal/utils.py", line 458, in <lambda>
    lambda: hydra.run(
  File "/home/check/miniconda3/envs/env_isaaclab/lib/python3.10/site-packages/hydra/_internal/hydra.py", line 132, in run
    _ = ret.return_value
  File "/home/check/miniconda3/envs/env_isaaclab/lib/python3.10/site-packages/hydra/core/utils.py", line 260, in return_value
    raise self._return_value
  File "/home/check/miniconda3/envs/env_isaaclab/lib/python3.10/site-packages/hydra/core/utils.py", line 186, in run_job
    ret.return_value = task_function(task_cfg)
  File "/home/check/Documents/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py", line 101, in hydra_main
    func(env_cfg, agent_cfg, *args, **kwargs)
  File "scripts/reinforcement_learning/rsl_rl/train.py", line 146, in main
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
  File "/home/check/miniconda3/envs/env_isaaclab/lib/python3.10/site-packages/rsl_rl/runners/on_policy_runner.py", line 208, in learn
    mean_value_loss, mean_surrogate_loss, mean_entropy, mean_rnd_loss, mean_symmetry_loss = self.alg.update()
  File "/home/check/miniconda3/envs/env_isaaclab/lib/python3.10/site-packages/rsl_rl/algorithms/ppo.py", line 251, in update
    self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
  File "/home/check/miniconda3/envs/env_isaaclab/lib/python3.10/site-packages/rsl_rl/modules/actor_critic.py", line 126, in act
    return self.distribution.sample()
  File "/home/check/miniconda3/envs/env_isaaclab/lib/python3.10/site-packages/torch/distributions/normal.py", line 84, in sample
    res = torch.normal(self.loc.expand(shape), std)
RuntimeError: normal expects all elements of std >= 0.0
2025-03-16 00:20:24 [129,028ms] [Warning] [omni.fabric.plugin] gFabricState->gUsdStageToSimStageWithHistoryMap had 1 outstanding SimStageWithHistory(s) at shutdown
2025-03-16 00:20:24 [129,362ms] [Warning] [carb] Recursive unloadAllPlugins() detected!
```

### Solutions
* fix the nan value to 0 or small value, failed!
* use small angle, 4096,1000(default)    failed! at 42th
* use big   angle, 1024,1000             failed! at 41th
* use big   angle, 512, 1000             failed! at 577th
* set the learning_rate to smaller       failed! 
* use small angle, 1024,1000             success            result not accurate
* franka         , 4096,1000(default)        success            result not accurate
* use big angle, damp 200                    success  just one time
* use big angle, limit v 0.01 and new seed   success!!!
* use big angle, limit v 0.01 and new seed   but failed at 170
* restart rtxdriver  like above  8000        failed at 110
* restart computer   like above  8000        failed at 244
* norestart          like above  2000        failed at <100
* pyoenix.py max_depenetration_velocity to 0.1 1000     success!!
