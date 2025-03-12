package main

import (
	"context"
	"io"
	"testing"

	"github.com/netbox-community/go-netbox/v4"
)

type Seed198 struct {
	Interfaces []*netbox.Interface
	Cable      *netbox.Cable
}

func (s *Seed198) Cleanup(t *testing.T, client *netbox.APIClient) {
	_, err := client.DcimAPI.DcimCablesDestroy(context.Background(), s.Cable.Id).Execute()
	if err != nil {
		t.Errorf("Failed to delete cable: %v", err)
	}
	for _, iface := range s.Interfaces {
		_, err := client.DcimAPI.DcimInterfacesDestroy(context.Background(), iface.Id).Execute()
		if err != nil {
			t.Errorf("Failed to delete interface: %v", err)
		}
	}
}
func HSeed198(t *testing.T, client *netbox.APIClient, defaults *Defaults) Seed198 {
	dr := netbox.BriefDeviceRequest{Name: defaults.Device.Name}

	interfaceRequests := []netbox.WritableInterfaceRequest{
		{
			Name:   randString(10),
			Device: dr,
			Type:   netbox.INTERFACETYPEVALUE__10GBASE_T,
		},
		{
			Name:   randString(10),
			Device: dr,
			Type:   netbox.INTERFACETYPEVALUE__10GBASE_T,
		},
	}

	interfaces := []*netbox.Interface{}
	for _, iface := range interfaceRequests {
		i, res, err := client.DcimAPI.DcimInterfacesCreate(context.Background()).WritableInterfaceRequest(iface).Execute()
		if err != nil {
			fatalHttp(t, "Failed to create interface", err, res)
		}
		body, _ := io.ReadAll(res.Body)
		t.Logf("Created interface: %s\n Theoretically: %d", string(body), i.Id)
		interfaces = append(interfaces, i)
	}

	cr := netbox.WritableCableRequest{
		ATerminations: []netbox.GenericObjectRequest{
			{
				ObjectType: "dcim.interface",
				ObjectId:   interfaces[0].Id,
			},
		},
		BTerminations: []netbox.GenericObjectRequest{
			{
				ObjectType: "dcim.interface",
				ObjectId:   interfaces[1].Id,
			},
		},
	}

	c, res, err := client.DcimAPI.DcimCablesCreate(context.Background()).WritableCableRequest(cr).Execute()
	if err != nil {
		fatalHttp(t, "Failed to created cable", err, res)
	}

	return Seed198{
		Interfaces: interfaces,
		Cable:      c,
	}
}

func Test198(t *testing.T) {
	client := HGetClient(t)
	defaults := HGetDefaults(t, client)
	seed := HSeed198(t, client, defaults)

	_, _, err := client.DcimAPI.DcimInterfacesTraceRetrieve(context.Background(), seed.Interfaces[0].Id).Execute()
	if err == nil {
		t.Errorf("Expected error, got nil")
	}

	seed.Cleanup(t, client)
}
