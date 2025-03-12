/*
*
# Error listing clusters with VirtualizationClustersList #199
```
> Hi, I have a problem with listing clusters:
>
> ```
> no value given for required property allocated_vcpus
> ```
>
> when running:
>
> ```
> res, status, err := c.VirtualizationAPI.VirtualizationClustersList(ctx).Limit(2).Execute()
> ```
>
> (status code 200)
>
> go-netbox: 4.2.2-1 netbox server: 4.1.10
```
*
*/
package main

import (
	"context"
	"testing"

	"github.com/netbox-community/go-netbox/v4"
)

type Seed199 struct {
	ClusterType *netbox.ClusterType
	Clusters    []*netbox.Cluster
}

func (s *Seed199) Cleanup(t *testing.T, client *netbox.APIClient) {
	t.Helper()
	for _, cluster := range s.Clusters {
		t.Logf("Deleting cluster %s", cluster.Name)
		res, err := client.VirtualizationAPI.VirtualizationClustersDestroy(context.Background(), cluster.GetId()).Execute()
		if err != nil {
			fatalHttp(t, "failed to delete cluster", err, res)
		}
	}
}

const Seed199Count = 10

func HSeed199(t *testing.T, client *netbox.APIClient) *Seed199 {
	t.Helper()
	seed := &Seed199{}
	seed.Clusters = []*netbox.Cluster{}

	ctr := netbox.ClusterTypeRequest{
		Name: randString(10),
		Slug: randString(10),
	}

	ct, res, err := client.VirtualizationAPI.VirtualizationClusterTypesCreate(context.Background()).ClusterTypeRequest(ctr).Execute()
	if err != nil {
		fatalHttp(t, "failed to create cluster type", err, res)
	}
	seed.ClusterType = ct

	for i := 0; i < Seed199Count; i++ {
		cluster := &netbox.WritableClusterRequest{
			Name: randString(10),
			Type: *netbox.NewBriefClusterTypeRequest(seed.ClusterType.Name, seed.ClusterType.Slug),
		}
		c, res, err := client.VirtualizationAPI.VirtualizationClustersCreate(context.Background()).WritableClusterRequest(*cluster).Execute()
		if err != nil {
			fatalHttp(t, "failed to create cluster", err, res)
		}
		seed.Clusters = append(seed.Clusters, c)
	}

	return seed
}

func Test199(t *testing.T) {
	harness := GetHarness(t)
	defer harness.Cleanup(t)
	client := harness.client

	seed := HSeed199(t, harness.client)
	harness.AddCleanup(seed)

	clusters, res, err := client.VirtualizationAPI.VirtualizationClustersList(context.Background()).Limit(2).Execute()
	if err != nil {
		fatalHttp(t, "failed to list clusters", err, res)
	}

	if len(clusters.Results) != 2 {
		t.Fatalf("expected 2 clusters, got %d", len(clusters.Results))
	}

}
