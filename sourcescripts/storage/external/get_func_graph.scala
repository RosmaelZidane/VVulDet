@main def exec(filename: String) = {
   importCode.python(filename)
   run.ossdataflow
   cpg.graph.E.map(node=>List(node.inNode.id, node.outNode.id, node.label, node.propertiesMap.get("VARIABLE"))).toJson |> filename + ".edges.json"
   cpg.graph.V.map(node=>node).toJson |> filename + ".nodes.json"
   delete
}
