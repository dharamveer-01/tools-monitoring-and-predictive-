class LoadBalancer:
    def __init__(self):
        self.load_distribution = {
            "S1": 33.3,
            "S2": 33.3,
            "S3": 33.3
        }
        # To avoid constant flip-flopping, only rebalance when states persist
    
    def redistribute(self, active_substations, health_data):
        """
        active_substations: list of sub_ids
        health_data: dict mapping sub_id -> {score, status}
        """
        if not active_substations:
            return self.load_distribution

        critical_subs = []
        healthy_subs = []

        for sub in active_substations:
            status = health_data.get(sub, {}).get("risk_level", "Healthy")
            if status == "Critical":
                critical_subs.append(sub)
            else:
                healthy_subs.append(sub)

        # If everyone's critical or everyone's healthy, maintain equal distribution of active nodes
        if len(critical_subs) == 0 or len(healthy_subs) == 0:
            target_load = 100.0 / len(active_substations)
            for sub in active_substations:
                self.load_distribution[sub] = target_load
            return self.load_distribution
        
        # We have a mix. Drop the critical ones to a safe minimal base (e.g. 10%)
        # and distribute remaining load among healthy ones.
        safe_critical_load = 10.0
        remaining_load = 100.0 - (len(critical_subs) * safe_critical_load)
        load_per_healthy = remaining_load / len(healthy_subs)

        for sub in critical_subs:
            self.load_distribution[sub] = safe_critical_load

        for sub in healthy_subs:
            self.load_distribution[sub] = load_per_healthy

        return self.load_distribution
