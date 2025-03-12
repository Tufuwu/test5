# Generated by Django 2.2.24 on 2021-06-21 13:39

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ("a4polls", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="OtherVote",
            fields=[
                (
                    "vote",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE,
                        primary_key=True,
                        related_name="other_vote",
                        serialize=False,
                        to="a4polls.Vote",
                    ),
                ),
                ("answer", models.CharField(max_length=250, verbose_name="Answer")),
            ],
        ),
        migrations.AddField(
            model_name="choice",
            name="is_other_choice",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="question",
            name="help_text",
            field=models.CharField(
                blank=True, max_length=250, verbose_name="Help text"
            ),
        ),
        migrations.AddField(
            model_name="question",
            name="is_open",
            field=models.BooleanField(default=False),
        ),
        migrations.CreateModel(
            name="Answer",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "created",
                    models.DateTimeField(
                        default=django.utils.timezone.now, editable=False
                    ),
                ),
                (
                    "modified",
                    models.DateTimeField(blank=True, editable=False, null=True),
                ),
                ("answer", models.CharField(max_length=750, verbose_name="Answer")),
                (
                    "creator",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    "question",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="answers",
                        to="a4polls.Question",
                    ),
                ),
            ],
            options={
                "ordering": ["id"],
                "unique_together": {("question", "creator")},
            },
        ),
    ]
