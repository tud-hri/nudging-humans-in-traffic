from lane import VLane, HLane


class IntersectionWorld:
    def __init__(self, dt: float, width: float, height: float):
        """
        The world our agents live in
        :param dt: simulation time step [second]
        :param width: world width [meter]
        :param height: world height [meter]
        """
        self.dt = dt
        self.width = width  # [m]
        self.height = height  # [m]

        self.agents = {}
        self.lanes = []

        self.create_intersection()

    def create_intersection(self):
        self.lanes.append(VLane([40., 0.], [40., 120.], 3.))
        self.lanes.append(VLane([37., 0.], [37., 120.], 3.))
        self.lanes.append(HLane([0., 30.], [40., 30.], 3.))

    def tick(self):
        # find action
        for _, agent in self.agents.items():
            agent.set_input()

        # apply action, integrate
        for _, agent, in self.agents.items():
            agent.tick()

    def draw(self, window, ppm):
        window.fill((33, 138, 33))

        for lane in self.lanes:
            lane.draw(window, ppm)

        # draw agents
        state_text = []
        for _, agent in self.agents.items():
            agent.draw(window, ppm)
            state_text.append(agent.text_state_render())

        pos = (5, 5)
        for txt in state_text:
            window.blit(txt, txt.get_rect(left=pos[0], top=pos[1]))
            pos = (pos[0], txt.get_rect().bottom + 0.25*txt.get_height())

