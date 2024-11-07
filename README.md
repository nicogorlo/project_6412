# 6.412 Robotic Manipulation - final project

## Students
- [Nitish Dashora](https://www.nitishdashora.com/)
- [Nicolas Gorlo](https://www.linkedin.com/in/nicolas-gorlo/)
- [Ben Zandonati](https://www.benzandonati.co.uk/)


## Project Description


## Installation

In the working directory run the following commands:

```bash
pip install -r requirements.txt 
```

```bash
pip install -e .
```

## Todo

- [x] Add point fingers
- [ ] change point finger positions 
- [ ] integrate point fingers to IK
- [ ] figure out way to model contact in drake (for our slightly compliant approach)
- [ ] GCS sets for point fingers and translation only

## Structure

- [ ] Environment class
    - [ ] setup
    - [ ] set initial state
    - [ ] visualize in meshcat
    - [ ] update_values (q1, q2, object pose)
    - [ ] query -> q1, q2, object pose
    - [ ] reset

- [ ] Planner
    - 