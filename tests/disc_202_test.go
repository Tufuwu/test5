/**

Issue Body:

```
Hi,

Below is my sample code.

// Set up NetBox API client
targetPrefix := "192.168.1.0/24" // Change to your desired prefix

client := netbox.NewNetboxWithAPIKey(netboxURL, apiToken)

// Step 1: Find the Prefix ID dynamically
prefixListReq := ipam.NewIpamPrefixesListParams().WithPrefix(&targetPrefix)
prefixListResp, err := client.Ipam.IpamPrefixesList(prefixListReq, nil)
if err != nil || prefixListResp.Payload == nil || len(prefixListResp.Payload.Results) == 0 {
fmt.Println(err)
fmt.Println(prefixListResp.Payload)
fmt.Println(prefixListResp.Payload.Results)
log.Fatalf("Prefix %s not found in NetBox", targetPrefix)
}
prefixID := prefixListResp.Payload.Results[0].ID

// Step 2: Allocate an Available IP from the Prefix
ipReq := ipam.NewIpamPrefixesAvailableIpsCreateParams().WithID(prefixID)
ipResp, err := client.Ipam.IpamPrefixesAvailableIpsCreate(ipReq, nil)
if err != nil || ipResp.Payload == nil || len(prefixListResp.Payload.Results) == 0 {
log.Fatalf("Failed to allocate IP: %v", err)
}

Though the IP address has been created in the netbox, verified via GUI, this API is throwing an error as below.
"Failed to allocate IP: json: cannot unmarshal object into Go value of type []*models.IPAddress"

Is there something wrong in the code above?
````
**/

package main

import (
	"context"
	"testing"

	"github.com/netbox-community/go-netbox/v4"
)

type Seed202Disc struct {
	Prefix *netbox.Prefix
	IP     *netbox.IPAddress
}

func (s *Seed202Disc) Cleanup(t *testing.T, client *netbox.APIClient) {
	t.Helper()

	res, err := client.IpamAPI.IpamPrefixesDestroy(context.Background(), s.Prefix.GetId()).Execute()
	if err != nil {
		fatalHttp(t, "failed to delete prefix", err, res)
	}

	res, err = client.IpamAPI.IpamIpAddressesDestroy(context.Background(), s.IP.GetId()).Execute()
	if err != nil {
		fatalHttp(t, "failed to delete ip address", err, res)
	}
}

func HSeed202Disc(t *testing.T, client *netbox.APIClient) *Seed202Disc {
	t.Helper()

	prefixReq := netbox.WritablePrefixRequest{
		Prefix: "192.168.1.0/24",
	}

	prefix, res, err := client.IpamAPI.IpamPrefixesCreate(context.Background()).WritablePrefixRequest(prefixReq).Execute()
	if err != nil {
		fatalHttp(t, "failed to create prefix", err, res)
	}

	ipReq := netbox.WritableIPAddressRequest{
		Address: "192.168.1.1/24",
	}

	ip, res, err := client.IpamAPI.IpamIpAddressesCreate(context.Background()).WritableIPAddressRequest(ipReq).Execute()
	if err != nil {
		fatalHttp(t, "failed to create ip address", err, res)
	}

	return &Seed202Disc{
		Prefix: prefix,
		IP:     ip,
	}
}

func Test202Disc(t *testing.T) {
	harness := GetHarness(t)
	defer harness.Cleanup(t)
	client := harness.client

	seed := HSeed202Disc(t, harness.client)
	harness.AddCleanup(seed)

	_, res, err := client.IpamAPI.IpamPrefixesList(context.Background()).Execute()
	if err != nil {
		fatalHttp(t, "failed to list prefixes", err, res)
	}

	ipAddressRequest := netbox.IPAddressRequest{
		Address: "192.168.1.1/24",
	}

	_, res, err = client.IpamAPI.IpamPrefixesAvailableIpsCreate(context.Background(), seed.Prefix.Id).IPAddressRequest([]netbox.IPAddressRequest{ipAddressRequest}).Execute()
	if err != nil {
		fatalHttp(t, "failed to create ip address", err, res)
	}
}
