"""Raw causal replay. Hidden regime parameters have no representation here."""
import numpy as np


class Replay:
    def __init__(self, capacity, obs_dim, history=16, seed=0,include_requested_hold=False):
        self.capacity, self.obs_dim, self.history = capacity, obs_dim, history
        self.rng = np.random.default_rng(seed)
        widths = dict(obs=obs_dim, next_obs=obs_dim, prev_action=6, action=6,
                      prev_reward=1, prev_h=1, r_src=1, R=1, h=1, next_h=1,
                      terminated=1, truncated=1, g=1, time=1, next_time=1)
        if include_requested_hold:widths["h_requested"]=1
        self.arrays = {k:np.zeros((capacity,n),np.float32) for k,n in widths.items()}
        self.episodes = np.full(capacity,-1,np.int64)
        self.serials = np.full(capacity,-1,np.int64)
        self.size, self.count = 0, 0
        self.excluded=set()
        self.pools={}
        self.pool_position=np.full(capacity,-1,np.int64)
        self.partition_cache=None

    def add(self, record, episode):
        if set(record) != set(self.arrays):
            raise ValueError(f"Public record fields differ: {set(record)^set(self.arrays)}")
        i = self.count % self.capacity
        if self.pool_position[i]>=0:
            pool=self.pools[float(self.arrays["g"][i,0])]
            position=int(self.pool_position[i]);last=pool.pop()
            if position<len(pool):pool[position]=last;self.pool_position[last]=position
            self.pool_position[i]=-1
        for k,v in record.items():
            self.arrays[k][i] = v
        self.episodes[i], self.serials[i] = episode, self.count
        if self.count not in self.excluded:
            pool=self.pools.setdefault(float(self.arrays["g"][i,0]),[])
            self.pool_position[i]=len(pool);pool.append(i)
        self.partition_cache=None
        self.count += 1
        self.size = min(self.count,self.capacity)

    def indices(self):
        return np.flatnonzero(self.serials >= max(0,self.count-self.capacity))

    def windows(self, indices):
        indices = np.asarray(indices,dtype=int)
        b, length, width = len(indices), self.history, self.obs_dim+8
        window = np.zeros((b,length,width),np.float32)
        for lag in range(length):
            ix = (indices-lag) % self.capacity
            valid = ((self.episodes[ix] == self.episodes[indices]) &
                     (self.serials[ix] == self.serials[indices]-lag) & (self.serials[ix]>=0))
            frame = np.concatenate([self.arrays[k][ix] for k in
                                    ("obs","prev_action","prev_reward","prev_h")],axis=1)
            window[valid,length-1-lag] = frame[valid]
        successor = np.concatenate((self.arrays["next_obs"][indices],self.arrays["action"][indices],
                                    self.arrays["r_src"][indices],self.arrays["h"][indices]),axis=1)
        next_window = np.concatenate((window[:,1:],successor[:,None]),axis=1)
        result = {k:v[indices] for k,v in self.arrays.items()}
        result.update(window=window,next_window=next_window)
        return result

    def sample(self, n, indices=None, balanced=True):
        if indices is None:
            pools={g:p for g,p in self.pools.items() if len(p)}
        else:
            if self.partition_cache is None or self.partition_cache[0] is not indices:
                selected=np.asarray(indices)
                pools={float(g):selected[self.arrays["g"][selected,0]==g].tolist()
                       for g in np.unique(self.arrays["g"][selected])}
                self.partition_cache=(indices,pools)
            pools=self.partition_cache[1]
        if not pools:
            raise ValueError("Empty replay partition")
        groups=list(pools)
        probability=None if balanced else np.asarray([len(pools[g]) for g in groups],float)/sum(map(len,pools.values()))
        tasks=self.rng.choice(groups,n,p=probability)
        selected=[pools[float(g)][int(self.rng.integers(len(pools[float(g)])))] for g in tasks]
        return self.windows(selected)

    def exclude_serials(self,serials):
        self.excluded=set(map(int,serials))
        self.pools={};self.pool_position[:]=-1;self.partition_cache=None
        for i in self.indices():
            if int(self.serials[i]) not in self.excluded:
                pool=self.pools.setdefault(float(self.arrays["g"][i,0]),[])
                self.pool_position[i]=len(pool);pool.append(int(i))

    def episode_split(self,strata=None):
        eligible = self.indices()
        episodes = np.unique(self.episodes[eligible])
        if strata is None and len(episodes)<5:
            raise ValueError("Calibration needs at least five complete/partial episodes for an episode split")
        if strata is None:
            validation = set(episodes[4::5].tolist())
        else:
            # Keep whole episodes together. A rare stratum with two to four
            # episodes contributes one validation episode and retains training
            # data. Larger strata retain the previous every-fifth convention.
            if any(int(e) not in strata for e in episodes):raise ValueError("Missing calibration episode stratum")
            groups={}
            for e in episodes:groups.setdefault(strata[int(e)],[]).append(int(e))
            validation=set()
            for group in groups.values():
                if len(group)>=2:validation.update(group[4::5] or group[-1:])
        val = np.array([i for i in eligible if self.episodes[i] in validation],dtype=int)
        train = np.array([i for i in eligible if self.episodes[i] not in validation],dtype=int)
        if not len(train) or not len(val):raise ValueError("Calibration needs distinct training and validation episodes")
        return train,val

    def state(self):
        return dict(arrays={k:v[:self.size].copy() for k,v in self.arrays.items()},
                    episodes=self.episodes[:self.size].copy(),serials=self.serials[:self.size].copy(),
                    size=self.size,count=self.count,capacity=self.capacity,obs_dim=self.obs_dim,
                    history=self.history,rng=self.rng.bit_generator.state,excluded=sorted(self.excluded))

    @classmethod
    def restore(cls,state, capacity=None):
        capacity = capacity or state["capacity"]
        if capacity < state["size"]:
            raise ValueError("New capacity cannot discard calibration records implicitly")
        if capacity != state["capacity"] and state["count"] > state["capacity"]:
            raise ValueError("Resizing a wrapped replay requires explicit chronological reindexing")
        obj = cls(capacity,state["obs_dim"],state["history"],include_requested_hold="h_requested" in state["arrays"])
        obj.size,obj.count = state["size"],state["count"]
        for k,v in state["arrays"].items():obj.arrays[k][:obj.size]=v
        obj.episodes[:obj.size],obj.serials[:obj.size]=state["episodes"],state["serials"]
        obj.rng.bit_generator.state=state["rng"]
        obj.exclude_serials(state.get("excluded",[]))
        return obj


def next_history(window, observation, action, received_reward, elapsed):
    frame = np.concatenate([np.asarray(observation),np.asarray(action),[received_reward,elapsed]]).astype(np.float32)
    return np.concatenate([window[1:],frame[None]],axis=0)


def initial_history(observation, length):
    window = np.zeros((length,len(observation)+8),np.float32)
    window[-1,:len(observation)] = observation
    return window
