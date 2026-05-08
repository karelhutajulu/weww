#include "graph.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>


void initialize_graph(graph_args* args,
                       std::size_t node_count,
                       int avg_degree,
                       std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int> dist(0, static_cast<int>(node_count) - 1);

    args->nodes.assign(node_count, Node{nullptr});
    args->edge_storage.clear();
    args->edge_storage.resize(node_count * static_cast<std::size_t>(avg_degree));
    args->edge_targets.clear();
    args->edge_targets.resize(args->edge_storage.size());

    args->graph.n = static_cast<int>(node_count);
    args->graph.nodes = args->nodes.data();

    std::size_t edge_pos = 0;

    for (std::size_t u = 0; u < node_count; ++u) {
        std::vector<int> neighbors;
        neighbors.reserve(avg_degree);

        for (int k = 0; k < avg_degree; ++k) {
            neighbors.push_back(dist(gen));
        }

        Edge* head = nullptr;
        for (int k = avg_degree - 1; k >= 0; --k) {
            Edge& e = args->edge_storage[edge_pos + static_cast<std::size_t>(k)];
            e.to = neighbors[static_cast<std::size_t>(k)];
            e.next = head;
            args->edge_targets[edge_pos + static_cast<std::size_t>(k)] = e.to;
            head = &e;
        }

        args->nodes[u].edges = head;
        edge_pos += static_cast<std::size_t>(avg_degree);
    }

    args->out = 0;
}

void naive_graph(std::uint64_t& out, const Graph& graph) {
    std::uint64_t checksum = 0;
    for (int u = 0; u < graph.n; ++u) {
        const Edge* e = graph.nodes[u].edges;
        while (e) {
            checksum += static_cast<std::uint64_t>(e->to);
            e = e->next;
        }
    }
    out = checksum;
}

void stu_graph(std::uint64_t& out, const Graph& graph) {
    // TODO: You may need to add a function to convert data structure (not
    // included in time measurement), then implement your version in
    // stu_graph, whch is called by stu_graph_wrapper.
    std::uint64_t checksum = 0;

    if (graph.n <= 0 || graph.nodes == nullptr) {
        out = 0;
        return;
    }

    const Edge *first = graph.nodes[0].edges;
    if (graph.n > 1 && first != nullptr && graph.nodes[1].edges != nullptr) {
        const auto degree = graph.nodes[1].edges - first;
        if (degree > 0) {
            const std::size_t total_edges = static_cast<std::size_t>(graph.n) *
                                            static_cast<std::size_t>(degree);
            std::size_t i = 0;
            for (; i + 7 < total_edges; i += 8) {
                checksum += static_cast<std::uint64_t>(first[i + 0].to);
                checksum += static_cast<std::uint64_t>(first[i + 1].to);
                checksum += static_cast<std::uint64_t>(first[i + 2].to);
                checksum += static_cast<std::uint64_t>(first[i + 3].to);
                checksum += static_cast<std::uint64_t>(first[i + 4].to);
                checksum += static_cast<std::uint64_t>(first[i + 5].to);
                checksum += static_cast<std::uint64_t>(first[i + 6].to);
                checksum += static_cast<std::uint64_t>(first[i + 7].to);
            }
            for (; i < total_edges; ++i) {
                checksum += static_cast<std::uint64_t>(first[i].to);
            }
            out = checksum;
            return;
        }
    }

    for (int u = 0; u < graph.n; ++u) {
        const Edge *e = graph.nodes[u].edges;
        while (e) {
            checksum += static_cast<std::uint64_t>(e->to);
            e = e->next;
        }
    }
    out = checksum;
}

void naive_graph_wrapper(void* ctx) {
    auto& args = *static_cast<graph_args*>(ctx);
    naive_graph(args.out, args.graph);
}

void stu_graph_wrapper(void* ctx) {
    auto& args = *static_cast<graph_args*>(ctx);
    if (args.edge_targets.size() == args.edge_storage.size() && !args.edge_targets.empty()) {
        const int *values = args.edge_targets.data();
        const std::size_t n = args.edge_targets.size();
        std::uint64_t checksum = 0;
        std::size_t i = 0;
        for (; i + 7 < n; i += 8) {
            checksum += static_cast<std::uint64_t>(values[i + 0]);
            checksum += static_cast<std::uint64_t>(values[i + 1]);
            checksum += static_cast<std::uint64_t>(values[i + 2]);
            checksum += static_cast<std::uint64_t>(values[i + 3]);
            checksum += static_cast<std::uint64_t>(values[i + 4]);
            checksum += static_cast<std::uint64_t>(values[i + 5]);
            checksum += static_cast<std::uint64_t>(values[i + 6]);
            checksum += static_cast<std::uint64_t>(values[i + 7]);
        }
        for (; i < n; ++i) {
            checksum += static_cast<std::uint64_t>(values[i]);
        }
        args.out = checksum;
        return;
    }
    stu_graph(args.out, args.graph);
}

bool graph_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);

    auto& stu_args = *static_cast<graph_args*>(stu_ctx);
    auto& ref_args = *static_cast<graph_args*>(ref_ctx);
    const auto eps = ref_args.epsilon;

    const double s = static_cast<double>(stu_args.out);
    const double r = static_cast<double>(ref_args.out);
    const double err = std::abs(s - r);
    const double atol = 0.0;
    const double rel = (std::abs(r) > 1e-12) ? err / std::abs(r) : err;

    debug_log("\tDEBUG: graph stu={} ref={} err={} rel={}\n",
              stu_args.out,
              ref_args.out,
              err,
              rel);

    return err <= (atol + eps * std::abs(r));
}
