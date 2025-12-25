# 🔍 Copilot Audit Prompts

> **One-shot master prompt** and **follow-up prompts** for comprehensive repo auditing.

---

## 🔧 Master Copilot Prompt (Full Audit)

**Paste this into VS Code Copilot Chat:**

```text
You are a senior robotics platform engineer and DevOps reviewer. Audit my entire Digital Twin Robotics Lab repo end-to-end and make it pass a production readiness gate.

Context (Architecture):
- Closed loop: Speech → Intent → ROS 2 goal → Nav2 path → Isaac Sim cmd_vel → sensor feedback → UI (RViz/Foxglove).
- Containers:
  A: Cognitive (NVIDIA Riva ASR + LLM via NIM) → emits JSON intents (e.g., {"action":"navigate","coordinates":[x,y]}).
  B: Control (ROS 2 Humble + Nav2 + Isaac ROS GEMs / AMCL).
  C: Simulation (Isaac Sim 4.2 headless with ROS 2 Bridge).
- Target OS: Ubuntu 20.04/22.04, GPU RTX 3080/4070+ (12GB VRAM+). Docker Compose deployment preferred.

**Your goals (produce concrete code/PRs, not just advice):**

1) Repo Mapping & Readability
   - Enumerate folders, packages, Dockerfiles, compose files, launch files, USD assets, scripts.
   - Generate a top-level README with architecture diagram (Markdown), quickstart, and operational runbook.
   - Add a CONTRIBUTING.md and ADRs (Architecture Decision Records) for key choices (Riva/NIM, Nav2 configs, Isaac ROS bridge, QoS).

2) Docker & Compose Validation
   - Validate docker-compose.yml: service names, networks, env, `gpus: all`, NVIDIA runtime, volumes, healthchecks, logging, restart policies, `.env` references.
   - Ensure Container A/B/C can resolve each other by service DNS names; document ports.
   - Add `hadolint` fixes to Dockerfiles, multi-stage builds, non-root users, pinned base images, cache-friendly layers.
   - Add `docker-compose config` and `compose up --build` steps in CI. Include artifacts: build logs and composed config snapshot.

3) Cognitive Service (Riva + LLM via NIM)
   - Verify Riva server startup, model bundles, VAD/punctuation settings, language packs, gRPC endpoints, TLS if applicable.
   - Validate Python client: retries, timeouts, structured logging, unit tests with sample audio and mocked gRPC.
   - Define a **Pydantic schema** for the JSON intent and validate all outputs. Add error-handling for empty/ambiguous intents and a fallback confirmation flow.
   - Stub/mock NIM LLM calls for tests; ensure intent extraction is deterministic for canned inputs.

4) ROS 2 Control Stack (Humble + Nav2 + Isaac ROS / AMCL)
   - Verify colcon workspace: `package.xml`, `CMakeLists.txt`, `ament` deps, `rosdep` keys, `setup.py` for Python nodes.
   - Run `ament_lint`, `ament_flake8`, `ament_cmake_cppcheck`, `clang-tidy` (C++), `ruff`, `black`, `mypy` (Python) and fix violations.
   - Inspect launch files: correct namespaces, `use_sim_time`, parameters for Nav2 (planner, controller, recovery behaviors).
   - Validate TF tree (base_link, odom, map), static transforms, URDF/Xacro correctness.
   - Enumerate topics and QoS (reliability/durability/depth). Ensure cmd_vel and sensor topics match bridge expectations.
   - Add `ros2 doctor`, `ros2 topic list`, `ros2 node list`, `ros2 bag record` smoke tests and snapshot logs to artifacts.

5) Isaac Sim (4.2) + ROS Bridge
   - Confirm USD scene loads headless, physics enabled, sensors defined (Lidar/Camera), ROS Bridge extension active.
   - Validate ROS domain ID, clock sync (`/clock`), frame IDs consistency, sensor rates within Nav2 expectations.
   - Add a headless test: spawn robot, publish a goal, ensure cmd_vel observed, pose updates, and target reached. Export a short MP4 or image frames for CI artifacts.

6) End-to-End Scenario Tests
   - Define at least 3 speech-driven scenarios:
     a) "Move the pallet to Zone B" (navigate to coordinates).
     b) "Robot, inspect the north shelf" (intent → navigation).
     c) Ambiguous command ("Go there") triggers clarification.
   - For each: record audio → ASR → intent → ROS goal → Nav2 path → sim motion → sensor feedback. Assert success/failure, time-to-goal, obstacle avoidance log entries.

7) Warnings-as-Errors & Clean Build
   - Make all builds treat warnings as errors (C++: `-Werror`, Python: `ruff` strict, mypy strict).
   - Ensure the build is clean: no warnings. If any remain, open PRs that resolve them.

8) CI/CD (GitHub Actions)
   - Add workflows:
     - `lint-test.yml`: pre-commit, linters, unit/integration tests.
     - `build-compose.yml`: build images (with `--pull`), run `docker-compose config`, smoke-run in headless mode.
     - `security.yml`: `trivy` image and filesystem scan, `bandit` for Python.
   - Cache colcon and pip; matrix for Ubuntu 20.04/22.04 and Python 3.10/3.11. Upload logs/artifacts (test reports, sim snapshots).
   - Gate PRs on CI green.

9) Developer UX
   - Provide VS Code devcontainer for seamless GPU dev (document enabling NVIDIA runtime).
   - Add `Makefile` / `Taskfile.yaml` with targets: `setup`, `compose-up`, `compose-down`, `lint`, `test`, `sim-smoke`, `ros2-diagnostics`.
   - Add Foxglove / RViz quickstart profiles and screenshots.

10) Deliverables
   - A structured **Audit Report** (Markdown): findings, severity, recommended fixes, PR links.
   - Auto-generated PRs with diffs for: Dockerfiles, compose, linters, CI, tests, ROS params/launch, Pydantic schema, Isaac headless test.
   - A final summary: zero warnings, green CI, deterministic scenario results.

Repo location: c:\Users\yamichai\OneDrive - Deloitte (O365D)\Documents\General\temp\digital-twin-robotics-lab

Start by reading the entire codebase. For every change, show the exact patch/diff and explain the reasoning in professional commit messages. Where runtime commands are needed, list them and include expected outputs/log snippets.
```

---

## 🔁 Follow-Up Prompts (Focused Passes)

### 1️⃣ Docker/Compose Deep Dive

```text
Review docker-compose.yml and all Dockerfiles in this repo. Enforce:
- GPU settings (`gpus: all` or deploy.resources.reservations.devices)
- Correct NVIDIA runtime configuration
- Healthchecks for all services
- Non-root users in containers
- Pinned base image tags (no :latest)
- Multi-stage builds to reduce image size
- Cache-friendly layer ordering

Run `docker-compose config` mentally and suggest fixes. Propose hadolint changes as exact diffs.
Show the exact file changes needed.
```

### 2️⃣ Riva + NIM Intent Schema

```text
Create a Pydantic model for intent payloads used between the Cognitive and Control layers. The schema should support:
- action: str (navigate, stop, status, inspect, etc.)
- target: Optional[str] (zone name)
- coordinates: Optional[dict] with x, y, theta
- confidence: float
- timestamp: datetime
- raw_transcript: str

Retrofit the ASR+LLM pipeline (cognitive_service/src/) to validate every output against this schema.
Add unit tests with sample audio transcripts and mocked NIM calls.
Show complete code changes and test files.
```

### 3️⃣ ROS 2 + Nav2 Diagnostics

```text
Generate a comprehensive ROS 2 diagnostics script that:
1. Runs `ros2 doctor` and captures output
2. Enumerates all nodes with `ros2 node list`
3. Lists all topics with `ros2 topic list`
4. Captures QoS settings for critical topics (/cmd_vel, /scan, /odom, /goal_pose)
5. Validates TF tree against expected frames (map, odom, base_link, base_footprint)
6. Generates a TF tree PDF using `ros2 run tf2_tools view_frames`
7. Checks Nav2 lifecycle states
8. Outputs a JSON report with pass/fail status

Add this script to CI as a smoke test and upload artifacts.
Show the complete Python script.
```

### 4️⃣ Isaac Sim Headless Smoke Test

```text
Create a headless Isaac Sim smoke test script that:
1. Loads the warehouse USD scene without GUI
2. Enables the ROS 2 Bridge extension
3. Spawns the robot at origin
4. Publishes a navigation goal to /goal_pose
5. Monitors /cmd_vel for movement commands
6. Tracks /odom for position updates
7. Asserts the robot reaches within 0.5m of target within 60 seconds
8. Captures simulation frames and saves as MP4 or image sequence
9. Exits with appropriate return code for CI

The script should work with Isaac Sim 4.2 and be runnable in CI.
Show the complete Python script.
```

### 5️⃣ Warnings-as-Errors Cleanup

```text
Make all code in this repo compile/lint with warnings-as-errors:

Python:
- Configure ruff with all rules enabled, treating warnings as errors
- Configure mypy in strict mode
- Configure black for formatting
- Fix ALL violations found

C++ (if any ROS nodes):
- Add -Werror to CMakeLists.txt
- Run clang-tidy and fix issues
- Run cppcheck and fix issues

Provide exact diffs for:
1. pyproject.toml configuration
2. All Python files that need fixes
3. CMakeLists.txt changes
4. Any C++ fixes needed

Commit message format: "fix(<scope>): resolve <linter> warning in <file>"
```

### 6️⃣ Security Audit

```text
Perform a security audit of this repo:

1. Run trivy filesystem scan - identify vulnerabilities in dependencies
2. Run bandit on all Python code - identify security issues
3. Check Dockerfiles for:
   - Running as root
   - Secrets in build args
   - Unpinned base images
   - Exposed ports without documentation
4. Check .env files for sensitive data exposure
5. Verify no hardcoded credentials in source code
6. Check for SQL injection, command injection, path traversal vulnerabilities

Generate a security report with:
- Severity (Critical/High/Medium/Low)
- Location (file:line)
- Description
- Recommended fix

Provide patches for all High and Critical issues.
```

### 7️⃣ End-to-End Test Scenarios

```text
Create comprehensive end-to-end test scenarios for the voice-controlled robot:

Scenario 1: "Navigate to loading dock"
- Input: Audio file or text transcript
- Expected: Intent parsed with action=navigate, target=loading_dock
- Expected: ROS goal published to Nav2
- Expected: Robot reaches (5.0, 2.0) within tolerance
- Assertions: Time < 30s, no collisions, path length reasonable

Scenario 2: "Stop immediately"
- Input: Stop command during navigation
- Expected: Intent parsed with action=stop
- Expected: Navigation cancelled
- Expected: Robot velocity goes to zero within 1s
- Assertions: cmd_vel.linear.x == 0, cmd_vel.angular.z == 0

Scenario 3: Ambiguous command "Go there"
- Input: Ambiguous transcript
- Expected: Intent confidence < 0.5
- Expected: System requests clarification
- Assertions: No navigation goal sent, error logged

Create pytest fixtures and test functions for each scenario.
Show complete test code.
```

### 8️⃣ CI/CD Pipeline Review

```text
Review and enhance the GitHub Actions CI/CD pipelines:

1. lint-test.yml:
   - Add pre-commit hooks check
   - Add ruff, black, mypy
   - Add pytest with coverage
   - Cache pip dependencies
   - Upload coverage report as artifact

2. build-compose.yml:
   - Build all Docker images
   - Run docker-compose config validation
   - Start services in detached mode
   - Run smoke tests
   - Capture logs on failure
   - Clean up resources

3. security.yml:
   - trivy filesystem scan
   - trivy image scan for built images
   - bandit Python security scan
   - Upload SARIF reports

Add matrix testing for:
- Ubuntu 20.04, 22.04
- Python 3.10, 3.11

Ensure all workflows:
- Have proper timeouts
- Cache appropriately
- Upload useful artifacts
- Have clear job names

Show complete workflow YAML files.
```

---

## 📋 Quick Reference Card

| Pass | Focus | Key Tools |
|------|-------|-----------|
| Master | Full audit | All tools |
| Docker | Containers | hadolint, trivy, compose |
| Schema | Data contracts | Pydantic, pytest |
| ROS 2 | Robot stack | ros2 cli, ament_lint |
| Isaac | Simulation | Isaac Sim Python API |
| Warnings | Code quality | ruff, mypy, clang-tidy |
| Security | Vulnerabilities | trivy, bandit |
| E2E | Integration | pytest, mock audio |
| CI/CD | Automation | GitHub Actions |

---

## 💡 Tips for Best Results

1. **Run Master Prompt First** - Get the full picture before diving deep
2. **Use @workspace** - Prefix prompts with `@workspace` in Copilot Chat
3. **Be Specific** - If master prompt is too broad, use focused follow-ups
4. **Iterate** - Run follow-ups multiple times, narrowing scope each time
5. **Review Diffs** - Always review proposed changes before accepting
6. **Test Locally** - Run suggested commands locally before CI
7. **Commit Incrementally** - Small, focused commits are easier to review

---

## 🎯 Expected Deliverables

After running the master prompt and follow-ups, you should have:

- [ ] Clean `docker-compose config` output
- [ ] All Dockerfiles passing `hadolint`
- [ ] Pydantic schemas for all inter-service messages
- [ ] ROS 2 diagnostics script in CI
- [ ] Isaac Sim headless smoke test
- [ ] Zero linter warnings (ruff, mypy, black)
- [ ] Security scan with no High/Critical issues
- [ ] E2E test suite with 3+ scenarios
- [ ] Full CI/CD pipeline with artifacts
- [ ] Updated documentation (README, CONTRIBUTING, ADRs)
