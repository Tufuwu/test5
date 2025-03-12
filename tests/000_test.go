package main

import (
	"os"
	"testing"

	"github.com/netbox-community/go-netbox/v4"
)

func GetHarness(t *testing.T) *Harness {
	t.Helper()
	client := HGetClient(t)
	defaults := HGetDefaults(t, client)
	return &Harness{
		client:   client,
		defaults: defaults,
	}
}

type Harness struct {
	client   *netbox.APIClient
	defaults *Defaults
	cleanup  []Cleanupable
}

func (h *Harness) AddCleanup(c Cleanupable) {
	h.cleanup = append(h.cleanup, c)
}

func (h *Harness) Cleanup(t *testing.T) {
	t.Helper()
	for _, c := range h.cleanup {
		c.Cleanup(t, h.client)
	}
	h.defaults.Cleanup(t, h.client)
}

type Cleanupable interface {
	Cleanup(t *testing.T, client *netbox.APIClient)
}

func HGetClient(t *testing.T) *netbox.APIClient {
	hostname := ""
	token := ""

	if os.Getenv("NETBOX_HOSTNAME") != "" {
		hostname = os.Getenv("NETBOX_HOSTNAME")
	}

	if os.Getenv("NETBOX_TOKEN") != "" {
		token = os.Getenv("NETBOX_TOKEN")
	}

	if hostname == "" || token == "" {
		t.Skip("NETBOX_HOSTNAME and NETBOX_TOKEN must be set")
	}

	client := netbox.NewAPIClientFor(hostname, token)
	return client
}
