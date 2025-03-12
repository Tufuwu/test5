/*
*
# Netbox tag requiring tagged_items on unmarshal while the API doesn't return that information #205
```
> Hello!
>
> https://github.com/netbox-community/go-netbox/blob/master/model_tag.go#L478
>
> The UnmarshalJSON method requires the `tagged_items` property in the JSON from netbox. When creating an object, that field is not shown:
>
>
>
>     {
>       "id" : 10,
>       "url" : "http://localhost:8001/api/extras/tags/10/",
>       "display_url" : "http://localhost:8001/extras/tags/10/",
>       "display" : "test-tag_basic-vu6hw8lmkf",
>       "name" : "test-tag_basic-vu6hw8lmkf",
>       "slug" : "test-tag_basic-2xheda2ay7",
>       "color" : "112233",
>       "description" : "This is a test",
>       "object_types" : [ ],
>       "created" : "2025-02-27T15:00:25.402848Z",
>       "last_updated" : "2025-02-27T15:00:25.402859Z"
>     }
>
>
>
>
>
>
>
> I used the following to create a tag:
>
>
>
>     api_res, _, err := client.ExtrasAPI.
>             ExtrasTagsCreate(ctx).
>             TagRequest(*tagRequest).Execute()
>
>
>
>
>
>
```
*
*/
package main

import (
	"context"
	"testing"

	"github.com/netbox-community/go-netbox/v4"
)

type Seed205 struct {
	Tags []*netbox.Tag
}

func (s *Seed205) Cleanup(t *testing.T, client *netbox.APIClient) {
	t.Helper()
	for _, tag := range s.Tags {
		t.Logf("Deleting tag %s", tag.Name)
		res, err := client.ExtrasAPI.ExtrasTagsDestroy(context.Background(), tag.GetId()).Execute()
		if err != nil {
			fatalHttp(t, "failed to delete tag", err, res)
		}
	}
}

func HSeed205(t *testing.T, client *netbox.APIClient) *Seed205 {
	t.Helper()
	seed := &Seed205{}
	seed.Tags = []*netbox.Tag{}

	// Create a tag
	tagRequest := &netbox.TagRequest{
		Name:        randString(10),
		Slug:        randString(10),
		Description: netbox.PtrString("This is a test"),
	}

	tag, res, err := client.ExtrasAPI.ExtrasTagsCreate(context.Background()).TagRequest(*tagRequest).Execute()
	if err != nil {
		fatalHttp(t, "failed to create tag", err, res)
	}
	seed.Tags = append(seed.Tags, tag)

	return seed
}

func Test205(t *testing.T) {
	harness := GetHarness(t)
	defer harness.Cleanup(t)
	client := harness.client

	seed := HSeed205(t, harness.client)
	harness.AddCleanup(seed)

	// Test creating a tag
	tagRequest := &netbox.TagRequest{
		Name:        randString(10),
		Slug:        randString(10),
		Description: netbox.PtrString("This is a test"),
	}

	tag, res, err := client.ExtrasAPI.ExtrasTagsCreate(context.Background()).TagRequest(*tagRequest).Execute()
	if err != nil {
		fatalHttp(t, "failed to create tag", err, res)
	}
	seed.Tags = append(seed.Tags, tag)

	// Test retrieving a tag
	retrievedTag, res, err := client.ExtrasAPI.ExtrasTagsRetrieve(context.Background(), tag.GetId()).Execute()
	if err != nil {
		fatalHttp(t, "failed to retrieve tag", err, res)
	}

	if retrievedTag.GetId() != tag.GetId() {
		t.Fatalf("expected tag ID %d, got %d", tag.GetId(), retrievedTag.GetId())
	}

	// Test listing tags
	tags, res, err := client.ExtrasAPI.ExtrasTagsList(context.Background()).Limit(10).Execute()
	if err != nil {
		fatalHttp(t, "failed to list tags", err, res)
	}

	if len(tags.Results) < 1 {
		t.Fatalf("expected at least 1 tag, got %d", len(tags.Results))
	}
}
