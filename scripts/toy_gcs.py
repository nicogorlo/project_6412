import numpy as np
from matplotlib import pyplot as plt
from gcs import construct_convex_hulls, construct_graph_of_convex_sets, generate_path

# define a few line segments for 1 contact mode (in 2D for now)
cm1 = [
    np.linspace([-5, 5], [0, 0], 5),
    np.linspace([-5, 0], [0, 5], 5),
]

cm2 = [
    np.linspace([-1, 5], [4, 0], 5),
    np.linspace([-1, 0], [4, 5], 5),
]
cm3 = [
    np.linspace([3, 5], [8, 0], 5),
    np.linspace([3, 0], [8, 5], 5),
]

cms = {
    "cm1": cm1,
    "cm2": cm2,
    "cm3": cm3,
}

chs = construct_convex_hulls(cms)
gcs = construct_graph_of_convex_sets(chs)
start = np.array([-4, 3])
end = np.array([7, 3])

print("graph edges:", [e.name() for e in gcs.Edges()])

edges, wps = generate_path(gcs, start, end)
print("Start", start)
print("End", end)
print("Waypoints", wps)
print("Edge names", [e.name() for e in edges])



# plot these line segments
for segment in cm1:
    plt.plot(segment[:, 0], segment[:, 1], 'red')
for segment in cm2:
    plt.plot(segment[:, 0], segment[:, 1], 'blue')
for segment in cm3:
    plt.plot(segment[:, 0], segment[:, 1], 'orange')

plt.plot(start[0], start[1], 'go')
plt.plot(end[0], end[1], 'go')

plt.ylim(-10, 10)
plt.xlim(-10, 10)

plt.show()
