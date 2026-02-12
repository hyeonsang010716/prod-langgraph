import strawberry
from strawberry.fastapi import GraphQLRouter

from app.api.graphql.resolvers import Query, Mutation

schema = strawberry.Schema(query=Query, mutation=Mutation)

graphql_router = GraphQLRouter(schema)
