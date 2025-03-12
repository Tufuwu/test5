/*

Issue Body:

```
Sup guyz! I'm tryna to get a devicelist like that
```
res1, httpRes, err := c.DcimAPI.DcimDevicesList(ctx).Status([]string{"active"}).Limit(10).Execute()

if err != nil {
    log.Printf("Error get devices list: %v", err)
}

log.Printf("%v", res1.Results)
log.Printf("Response: %v\n", res1)
log.Printf("HTTP Response: %+v", httpRes)
```
but all i got is
```
2024/09/06 17:13:41 Error get devices list: no value given for required property device_count
2024/09/06 17:13:41 []
2024/09/06 17:13:41 Response: &{0 {<nil> false} {<nil> false} [] map[]}
2024/09/06 17:13:41 HTTP Response: &{Status:200 OK StatusCode:200 Proto:HTTP/1.1 ProtoMajor:1 ProtoMinor:1 Header:map[Allow:[GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS] Connection:[keep-alive] Content-L
anguage:[en] Content-Length:[5577] Content-Type:[application/json] Cross-Origin-Opener-Policy:[same-origin] Date:[Fri, 06 Sep 2024 17:13:41 GMT] Referrer-Policy:[same-origin] Server:[nginx] Vary:[HX-Reque
st, Accept-Language, Cookie, origin] X-Content-Type-Options:[nosniff] X-Frame-Options:[SAMEORIGIN] X-Request-Id:[7fed8b0e-c509-450d-9d5f-23ef54cf2c2b]] Body:{Reader:{"count":3,"next":null,"previous":null,
"results":[{ ___HERE ALL MY DEVICES___ }]}} ContentLength:5577 TransferEncoding:[] Close:false Uncompressed:false Trailer:map[] Request:0xc0001712c0 TLS:<nil>}
```

Why the list is empty? Am i doing something wrong or it's just because of bugs: https://github.com/netbox-community/netbox/issues/16670 https://github.com/netbox-community/go-netbox/issues/165

P.S. I use the latest netbox 4.1.0
````


*/

package main

import (
	"context"
	"testing"

	"github.com/netbox-community/go-netbox/v4"
)

type Seed183Disc struct {
	Devices []*netbox.DeviceWithConfigContext
}

func (s *Seed183Disc) Cleanup(t *testing.T, client *netbox.APIClient) {
	t.Helper()

	for _, device := range s.Devices {
		res, err := client.DcimAPI.DcimDevicesDestroy(context.Background(), device.Id).Execute()
		if err != nil {
			fatalHttp(t, "failed to delete device", err, res)
		}
	}
}

const disc183DeviceCount = 10

var disc183DeviceDescription = "test-disc-183"

func HSeed183Disc(t *testing.T, client *netbox.APIClient, defaults *Defaults) *Seed183Disc {
	t.Helper()
	seed := &Seed183Disc{}

	for i := 0; i < disc183DeviceCount; i++ {
		name := randString(10)
		device := netbox.WritableDeviceWithConfigContextRequest{
			Name:        *netbox.NewNullableString(&name),
			Description: &disc183DeviceDescription,
			Site:        *netbox.NewBriefSiteRequest(defaults.Site.Name, defaults.Site.Slug),
			DeviceType:  *netbox.NewBriefDeviceTypeRequest(*netbox.NewBriefManufacturerRequest(defaults.Manufacturer.Name, defaults.Manufacturer.Slug), defaults.DeviceType.Model, defaults.DeviceType.Slug),
			Role:        *netbox.NewBriefDeviceRoleRequest(defaults.DeviceRole.Name, defaults.DeviceRole.Slug),
		}

		d, res, err := client.DcimAPI.DcimDevicesCreate(context.Background()).WritableDeviceWithConfigContextRequest(device).Execute()
		if err != nil {
			fatalHttp(t, "failed to create device", err, res)
		}
		seed.Devices = append(seed.Devices, d)
	}

	return seed
}

func Test183Disc(t *testing.T) {
	harness := GetHarness(t)
	defer harness.Cleanup(t)
	client := harness.client

	seed := HSeed183Disc(t, client, harness.defaults)
	harness.AddCleanup(seed)

	devicelist, res, err := client.DcimAPI.DcimDevicesList(context.Background()).Description([]string{disc183DeviceDescription}).Execute()
	if err != nil {
		fatalHttp(t, "failed to get device list", err, res)
	}

	if len(devicelist.Results) != disc183DeviceCount {
		t.Fatalf("expected %d devices, got %d", disc183DeviceCount, len(devicelist.Results))
	}
}
