import collections
import dataclasses
import json
import logging
import math
import pathlib
import time

import imageio
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    scenario: str = ""  # Project scenario: push_pi or push_p_i. Empty keeps the upstream benchmark path.
    duration_seconds: int = 30  # Policy-action time for project scenarios.
    smoke: bool = False  # Run project scenarios for 6 seconds instead of duration_seconds.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    if args.scenario:
        _eval_push_pi(args)
        return

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    element, img = _policy_element(obs, str(task_description), args.resize_size)

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _policy_element(obs, prompt: str, resize_size: int):
    # IMPORTANT: rotate 180 degrees to match LIBERO training preprocessing.
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize_size, resize_size))
    return {
        "observation/image": image,
        "observation/wrist_image": wrist,
        "observation/state": np.concatenate(
            (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
        "prompt": prompt,
    }, image


def _eval_push_pi(args: Args) -> None:
    from examples.libero.push_pi_env import CONTROL_HZ
    from examples.libero.push_pi_env import create_env

    seconds = 6 if args.smoke else args.duration_seconds
    if not 1 <= seconds <= 300:
        raise ValueError("duration_seconds must be between 1 and 300")
    policy_steps = seconds * CONTROL_HZ
    env, prompt = create_env(
        args.scenario,
        resolution=LIBERO_ENV_RESOLUTION,
        seed=args.seed,
        horizon=policy_steps + args.num_steps_wait + 1,
    )
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    output = pathlib.Path(args.video_out_path)
    output.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    video_path = output / f"{stamp}_{args.scenario}_{seconds}s.mp4"
    result_path = video_path.with_suffix(".json")
    action_plan = collections.deque()
    frames = []
    latencies_ms = []
    sticky_success = False
    best_coverage = 0.0
    final_coverage = {}
    try:
        obs = env.reset()
        for _ in range(args.num_steps_wait):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        for _ in tqdm.tqdm(range(policy_steps), desc=args.scenario):
            element, image = _policy_element(obs, prompt, args.resize_size)
            frames.append(image)
            if not action_plan:
                started = time.perf_counter_ns()
                action_chunk = np.asarray(client.infer(element)["actions"])
                latencies_ms.append((time.perf_counter_ns() - started) / 1_000_000)
                if action_chunk.ndim != 2 or action_chunk.shape[1] != 7 or len(action_chunk) < args.replan_steps:
                    raise ValueError("LIBERO policy actions must be a finite chunk with shape (N, 7)")
                if not np.issubdtype(action_chunk.dtype, np.floating) or not np.isfinite(action_chunk).all():
                    raise ValueError("LIBERO policy actions must be finite floating values")
                action_plan.extend(action_chunk[: args.replan_steps])
            action = np.asarray(action_plan.popleft())
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError("applied LIBERO action must be finite with shape (7,)")
            obs, _, done, _ = env.step(action.tolist())
            final_coverage = env.env.coverage()
            best_coverage = max(best_coverage, final_coverage["overall"])
            sticky_success = sticky_success or bool(done)
    finally:
        client.close()
        env.close()

    imageio.mimwrite(video_path, frames, fps=CONTROL_HZ)
    result = {
        "scenario": args.scenario,
        "seed": args.seed,
        "policy_steps": len(frames),
        "policy_seconds": len(frames) / CONTROL_HZ,
        "control_hz": CONTROL_HZ,
        "settle_steps": args.num_steps_wait,
        "success": sticky_success,
        "best_coverage": best_coverage,
        "final_coverage": final_coverage,
        "policy_requests": len(latencies_ms),
        "policy_latency_ms": {
            "mean": float(np.mean(latencies_ms)),
            "p95": float(np.percentile(latencies_ms, 95)),
        },
        "video": str(video_path),
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    from libero.libero import get_libero_path

    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
