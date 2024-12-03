# 6.4212 Robotic Manipulation - final project

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

## Thoughts:
- Conceptual question: We are creating a couple of sets per contact mode. In the case that several sets of different contact modes overlap, how do we prevent the method from arbitrarily switching contact modes?
    we need to somehow add an additional cost for switching contact modes
- Computational complexity of computing convex hull contains relates to lower dimensionality of purely tabletop problem: Each point is basically also on the edge of the convex hull if the volume is really small.