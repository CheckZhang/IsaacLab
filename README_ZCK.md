### train an envirment: 
```
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Reach-Franka-v0 --headless
```

### play an envirment:
```
./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py --task Isaac-Reach-Franka-v0 --num_envs 32 --load_run run_folder_name --checkpoint model.pt
```

### record video(.mp4):
```
./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py --task Isaac-Velocity-Rough-H1-v0 --headless --video --video_length 200
```


### debug an envirment:
set launch.json file:
```
        {
            "name": "H1 train",
            "type": "python",
            "request": "launch",
            "args": [
                "--task",
                "Isaac-Velocity-Rough-H1-v0",
                "--headless"
            ],
            "program": "${workspaceFolder}/source/standalone/workflows/rsl_rl/train.py",
            "console": "integratedTerminal"
        }
```

### ISAACLAB_NUCLEUS_DIR: http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.2/Isaac/IsaacLab

### SIM DIR: ~/miniconda3/envs/isaaclab/lib/python3.10/site-packages/isaacsim