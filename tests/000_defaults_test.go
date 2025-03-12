package main

import (
	"context"
	"io"
	"net/http"
	"testing"

	"math/rand"

	"github.com/netbox-community/go-netbox/v4"
)

const (
	manufacturerName = "test-manufacturer"
	deviceTypeName   = "test-device-type"
	siteName         = "test-site"
	deviceName       = "test-device"
	deviceRoleName   = "test-device-role"
	interfaceName    = "test-interface"
)

type Defaults struct {
	Site         netbox.Site
	DeviceRole   netbox.DeviceRole
	Manufacturer netbox.Manufacturer
	DeviceType   netbox.DeviceType
	Device       netbox.DeviceWithConfigContext
}

func (d *Defaults) Cleanup(t *testing.T, client *netbox.APIClient) {

	_, err := client.DcimAPI.DcimDevicesDestroy(context.Background(), d.Device.Id).Execute()
	if err != nil {
		t.Errorf("Failed to delete device: %v", err)
	}

	_, err = client.DcimAPI.DcimDeviceTypesDestroy(context.Background(), d.DeviceType.Id).Execute()
	if err != nil {
		t.Errorf("Failed to delete device type: %v", err)
	}

	_, err = client.DcimAPI.DcimManufacturersDestroy(context.Background(), d.Manufacturer.Id).Execute()
	if err != nil {
		t.Errorf("Failed to delete manufacturer: %v", err)
	}

	_, err = client.DcimAPI.DcimDeviceRolesDestroy(context.Background(), d.DeviceRole.Id).Execute()
	if err != nil {
		t.Errorf("Failed to delete device role: %v", err)
	}

	_, err = client.DcimAPI.DcimSitesDestroy(context.Background(), d.Site.Id).Execute()
	if err != nil {
		t.Errorf("Failed to delete site: %v", err)
	}
}

func randString(n int) string {
	letters := []rune("abcdefghijklmnopqrstuvwxyz")
	s := make([]rune, n)
	for i := range s {
		s[i] = letters[rand.Intn(len(letters))]
	}
	return string(s)
}

func fatalHttp(t *testing.T, msg string, err error, res *http.Response) {
	if res == nil {
		t.Fatalf("%s: %v", msg, err)
		return
	}
	body, _ := io.ReadAll(res.Body)
	t.Fatalf("%s: %v\n response: %v", msg, err, string(body))
}

func HGetDefaults(t *testing.T, client *netbox.APIClient) *Defaults {
	site := netbox.WritableSiteRequest{
		Name: siteName + randString(4),
		Slug: siteName + randString(4),
	}

	newSite, res, err := client.DcimAPI.DcimSitesCreate(context.Background()).WritableSiteRequest(site).Execute()
	if err != nil {
		fatalHttp(t, "Failed to create site", err, res)
	}

	deviceRole := netbox.DeviceRoleRequest{
		Name: deviceRoleName + randString(4),
		Slug: deviceRoleName + randString(4),
	}

	newDeviceRole, res, err := client.DcimAPI.DcimDeviceRolesCreate(context.Background()).DeviceRoleRequest(deviceRole).Execute()
	if err != nil {
		fatalHttp(t, "Failed to create device role", err, res)
	}

	manufacturer := netbox.ManufacturerRequest{
		Name: manufacturerName + randString(4),
		Slug: manufacturerName + randString(4),
	}

	newManufacturer, res, err := client.DcimAPI.DcimManufacturersCreate(context.Background()).ManufacturerRequest(manufacturer).Execute()
	if err != nil {
		fatalHttp(t, "Failed to create manufacturer", err, res)
	}

	deviceType := netbox.WritableDeviceTypeRequest{
		Model:        deviceTypeName + randString(4),
		Slug:         deviceTypeName + randString(4),
		Manufacturer: *netbox.NewBriefManufacturerRequest(newManufacturer.Name, newManufacturer.Slug),
	}

	newDeviceType, res, err := client.DcimAPI.DcimDeviceTypesCreate(context.Background()).WritableDeviceTypeRequest(deviceType).Execute()
	if err != nil {
		fatalHttp(t, "Failed to create device type", err, res)
	}

	name := deviceName + randString(4)

	device := netbox.WritableDeviceWithConfigContextRequest{
		Name:       *netbox.NewNullableString(&name),
		Role:       *netbox.NewBriefDeviceRoleRequest(newDeviceRole.Name, newDeviceRole.Slug),
		Site:       *netbox.NewBriefSiteRequest(newSite.Name, newSite.Slug),
		DeviceType: *netbox.NewBriefDeviceTypeRequest(*netbox.NewBriefManufacturerRequest(newManufacturer.Name, newManufacturer.Slug), newDeviceType.Model, newDeviceType.Slug),
	}

	newDevice, res, err := client.DcimAPI.DcimDevicesCreate(context.Background()).WritableDeviceWithConfigContextRequest(device).Execute()
	if err != nil {
		fatalHttp(t, "Failed to create device", err, res)
	}

	return &Defaults{
		Site:         *newSite,
		DeviceRole:   *newDeviceRole,
		Device:       *newDevice,
		Manufacturer: *newManufacturer,
		DeviceType:   *newDeviceType,
	}
}

func TestDefaults(t *testing.T) {

	client := HGetClient(t)

	defaults := HGetDefaults(t, client)

	defaults.Cleanup(t, client)
}
