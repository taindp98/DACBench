import pddl
import sccs
import timers

from collections import defaultdict


DEBUG = False


def handle_axioms(operators, axioms, goals):
    axioms_by_atom = get_axioms_by_atom(axioms)

    axiom_literals = compute_necessary_axiom_literals(axioms_by_atom, operators, goals)
    axiom_init = get_axiom_init(axioms_by_atom, axiom_literals)
    with timers.timing("Simplifying axioms"):
        axioms = simplify_axioms(axioms_by_atom, axiom_literals)
    axioms = compute_negative_axioms(axioms_by_atom, axiom_literals)
    # NOTE: compute_negative_axioms more or less invalidates axioms_by_atom.
    #       Careful with that axe, Eugene!
    axiom_layers = compute_axiom_layers(axioms, axiom_init)
    if DEBUG:
        verify_layering_condition(axioms, axiom_init, axiom_layers)
    return axioms, list(axiom_init), axiom_layers


def verify_layering_condition(axioms, axiom_init, axiom_layers):
    # This function is only used for debugging.
    variables_in_heads = set()
    literals_in_heads = set()
    variables_with_layers = set()

    for axiom in axioms:
        head = axiom.effect
        variables_in_heads.add(head.positive())
        literals_in_heads.add(head)
    variables_with_layers = set(axiom_layers.keys())

    # 1. Each derived variable only appears in heads with one
    #    polarity, i.e., never positively *and* negatively.
    if False:
        print("Verifying 1...")
        for literal in literals_in_heads:
            assert literal.negate() not in literals_in_heads, literal
    else:
        print("Verifying 1... [skipped]")
        # We currently violate this condition because we introduce
        # "negated axioms". See issue454 and issue453.

    # 2. A variable has a defined layer iff it appears in a head.
    #    (This is stricter than it needs to be; we could allow
    #    derived variables that are never generated by a rule.
    #    But this test follows the axiom simplification step, and
    #    after simplification this should not be too strict.)
    #    All layers are integers and at least 0.
    #    (Note: the "-1" layer for non-derived variables is
    #    set elsewhere.)
    print("Verifying 2...")
    assert variables_in_heads == variables_with_layers
    for atom, layer in axiom_layers.items():
        assert isinstance(layer, int)
        assert layer >= 0

    # 3. For every derived variable, it occurs in axiom_init iff
    #    its negation occurs as the head of an axiom.
    if False:
        print("Verifying 3...")
        for init in list(axiom_init):
            assert init.negate() in literals_in_heads
        for literal in literals_in_heads:
            assert (literal.negated) == (literal.positive() in axiom_init)
    else:
        print("Verifying 3 [weaker version]...")
        # We currently violate this condition because we introduce
        # "negated axioms". See issue454 and issue453.
        #
        # The weaker version we test here is "For every derived variable:
        # [it occurs in axiom_init iff its negation occurs as the
        # head of an axiom] OR [it occurs with both polarities in
        # heads of axioms]."
        for init in list(axiom_init):
            assert init.negate() in literals_in_heads
        for literal in literals_in_heads:
            assert (literal.negated) == (literal.positive() in axiom_init) or (
                literal.negate() in literals_in_heads
            )

    # 4. For every rule head <- ... cond ... where cond is a literal
    #    of a derived variable where the layer of head is equal to
    #    the layer of cond, cond occurs with the same polarity in heads.
    #
    # Note regarding issue454 and issue453: Because of the negated axioms
    # mentioned in these issues, a derived variable may appear with *both*
    # polarities in heads. This makes this test less strong than it would
    # be otherwise. When these issues are addressed and axioms only occur
    # with one polarity in heads, this test will remain correct in its
    # current form, but it will be able to detect more violations of the
    # layering property.
    print("Verifying 4...")
    for axiom in axioms:
        head = axiom.effect
        head_positive = head.positive()
        body = axiom.condition
        for cond in body:
            cond_positive = cond.positive()
            if (
                cond_positive in variables_in_heads
                and axiom_layers[cond_positive] == axiom_layers[head_positive]
            ):
                assert cond in literals_in_heads

    # 5. For every rule head <- ... cond ... where cond is a literal
    #    of a derived variable, the layer of head is greater or equal
    #    to the layer of cond.
    print("Verifying 5...")
    for axiom in axioms:
        head = axiom.effect
        head_positive = head.positive()
        body = axiom.condition
        for cond in body:
            cond_positive = cond.positive()
            if cond_positive in variables_in_heads:
                # We need the assertion to be on a single line for
                # our error handler to be able to print the line.
                assert axiom_layers[cond_positive] <= axiom_layers[head_positive], (
                    axiom_layers[cond_positive],
                    axiom_layers[head_positive],
                )


def get_axioms_by_atom(axioms):
    axioms_by_atom = {}
    for axiom in axioms:
        axioms_by_atom.setdefault(axiom.effect, []).append(axiom)
    return axioms_by_atom


def compute_axiom_layers(axioms, axiom_init):
    # We include this assertion to make sure testing membership in
    # axiom_init is efficient.
    assert isinstance(axiom_init, set)

    # Collect all atoms for derived variables.
    derived_atoms = set()
    for axiom in axioms:
        head_atom = axiom.effect.positive()
        derived_atoms.add(head_atom)

    # Collect dependencies between derived variables:
    # 1. "u depends on v" if there is an axiom with variable u
    #    in the head and variable v in the body.
    # 2. "u NBF-depends on v" if additionally the value with which
    #    v occurs in the body is its NBF (negation-by-failure) value.
    #
    # We represent depends_on as a dictionary mapping each "u" to
    # the list of "v"s such that u depends on v. Note that we do not
    # use a defaultdict because the SCC finding algorithm requires
    # that all nodes are present as keys in the dict, even if they
    # have no successors.
    #
    # We do not represent NBF-depends on independently, but we do keep
    # of a set of triples "weighted_depends_on" which contains all
    # triples (u, v, weight) representing dependencies from u to v,
    # where weight is 1 for NBF dependencies and 0 for other
    # dependencies. Each such triple represents the constraint
    # layer(u) >= layer(v) + weight.
    depends_on = dict((u, []) for u in derived_atoms)
    weighted_depends_on = set()
    for axiom in axioms:
        if (
            axiom.effect in axiom_init
            or axiom.effect.negated
            and axiom.effect.positive() not in axiom_init
        ):
            # Skip axioms whose head is the negation-by-failure value.
            # These are redundant axioms that should eventually go away
            # or at least have some kind of special status that marks
            # them as "not the primary axioms".
            continue
        u = axiom.effect.positive()
        for condition in axiom.condition:
            v = condition.positive()
            if v in derived_atoms:
                v_polarity = not condition.negated
                v_init_polarity = v in axiom_init
                # TODO: Don't include duplicates in depends_on.
                depends_on[u].append(v)
                if v_polarity == v_init_polarity:
                    weight = 1
                else:
                    weight = 0
                weighted_depends_on.add((u, v, weight))

    # Compute the SCCs of dependencies according to depends_on,
    # in topological order.
    atom_sccs = sccs.get_sccs_adjacency_dict(depends_on)

    # Compute an index mapping each atom to the id of its SCC.
    atom_to_scc_id = {}
    for scc in atom_sccs:
        scc_id = id(scc)
        for atom in scc:
            atom_to_scc_id[atom] = scc_id

    # Compute a weighted digraph representing the dependencies
    # between SCCs. SCCs U and V are represented by their IDs.
    # - We have id(V) in scc_weighted_depends_on[id(U)] iff
    #   some variable u in U depends on some variable v in V.
    # - If there is a dependency, scc_weighted_depends_on[id(U)][id(V)]
    #   is the weight of the dependency: +1 if an NBF-dependency
    #   exists, 0 otherwise.
    # We want the digraph to be acyclic and hence ignore self-loops.
    # A self-loop of weight 1 indicates non-stratifiability.
    scc_weighted_depends_on = defaultdict(dict)
    for u, v, weight in weighted_depends_on:
        scc_u_id = atom_to_scc_id[u]
        scc_v_id = atom_to_scc_id[v]
        if scc_u_id == scc_v_id:
            # Ignore self-loops unless they are self-loops based on
            # NBF dependencies, which occur iff the axioms are
            # non-stratifiable.
            if weight == 1:
                raise ValueError("Cyclic dependencies in axioms; cannot stratify.")
        else:
            old_weight = scc_weighted_depends_on[scc_u_id].get(scc_v_id, -1)
            if weight > old_weight:
                scc_weighted_depends_on[scc_u_id][scc_v_id] = weight

    # The layer of variable u is the longest path (taking into account
    # the weights) in the weighted digraph defined by
    # scc_weighted_depends_on from the SCC of u to any sink.

    # We first compute the longest paths in the SCC digraph. This
    # computation exploits that atom_sccs is given in
    # topological sort order.
    scc_id_to_layer = {}
    for scc in reversed(atom_sccs):
        scc_id = id(scc)
        layer = 0
        for succ_scc_id, weight in scc_weighted_depends_on[scc_id].items():
            layer = max(layer, scc_id_to_layer[succ_scc_id] + weight)
        scc_id_to_layer[scc_id] = layer

    # Finally, we set the layers for all nodes based on the layers of
    # their SCCs.
    layers = {}
    for scc in atom_sccs:
        scc_layer = scc_id_to_layer[id(scc)]
        for atom in scc:
            layers[atom] = scc_layer

    # for atom, layer in layers.items():
    #    print("Layer %d: %s" % (layer, atom))
    return layers


def compute_necessary_axiom_literals(axioms_by_atom, operators, goal):
    necessary_literals = set()
    queue = []

    def register_literals(literals, negated):
        for literal in literals:
            if literal.positive() in axioms_by_atom:  # This is an axiom literal
                if negated:
                    literal = literal.negate()
                if literal not in necessary_literals:
                    necessary_literals.add(literal)
                    queue.append(literal)

    # Initialize queue with axioms required for goal and operators.
    register_literals(goal, False)
    for op in operators:
        register_literals(op.precondition, False)
        for (cond, _) in op.add_effects:
            register_literals(cond, False)
        for (cond, _) in op.del_effects:
            register_literals(cond, True)

    while queue:
        literal = queue.pop()
        axioms = axioms_by_atom[literal.positive()]
        for axiom in axioms:
            register_literals(axiom.condition, literal.negated)
    return necessary_literals


def get_axiom_init(axioms_by_atom, necessary_literals):
    result = set()
    for atom in axioms_by_atom:
        if atom not in necessary_literals and atom.negate() in necessary_literals:
            # Initial value for axiom: False (which is omitted due to closed world
            # assumption) unless it is only needed negatively.
            result.add(atom)
    return result


def simplify_axioms(axioms_by_atom, necessary_literals):
    necessary_atoms = set([literal.positive() for literal in necessary_literals])
    new_axioms = []
    for atom in necessary_atoms:
        axioms = simplify(axioms_by_atom[atom])
        axioms_by_atom[atom] = axioms
        new_axioms += axioms
    return new_axioms


def remove_duplicates(alist):
    next_elem = 1
    for i in range(1, len(alist)):
        if alist[i] != alist[i - 1]:
            alist[next_elem] = alist[i]
            next_elem += 1
    alist[next_elem:] = []


def simplify(axioms):
    """Remove duplicate axioms, duplicates within axioms, and dominated axioms."""

    # Remove duplicates from axiom conditions.
    for axiom in axioms:
        axiom.condition.sort()
        remove_duplicates(axiom.condition)

    # Remove dominated axioms.
    axioms_to_skip = set()
    axioms_by_literal = {}
    for axiom in axioms:
        if axiom.effect in axiom.condition:
            axioms_to_skip.add(id(axiom))
        else:
            for literal in axiom.condition:
                axioms_by_literal.setdefault(literal, set()).add(id(axiom))

    for axiom in axioms:
        if id(axiom) in axioms_to_skip:
            continue  # Required to keep one of multiple identical axioms.
        if not axiom.condition:  # empty condition: dominates everything
            return [axiom]
        literals = iter(axiom.condition)
        dominated_axioms = axioms_by_literal[next(literals)]
        for literal in literals:
            dominated_axioms &= axioms_by_literal[literal]
        for dominated_axiom in dominated_axioms:
            if dominated_axiom != id(axiom):
                axioms_to_skip.add(dominated_axiom)
    return [axiom for axiom in axioms if id(axiom) not in axioms_to_skip]


def compute_negative_axioms(axioms_by_atom, necessary_literals):
    new_axioms = []
    for literal in necessary_literals:
        if literal.negated:
            new_axioms += negate(axioms_by_atom[literal.positive()])
        else:
            new_axioms += axioms_by_atom[literal]
    return new_axioms


def negate(axioms):
    assert axioms
    result = [pddl.PropositionalAxiom(axioms[0].name, [], axioms[0].effect.negate())]
    for axiom in axioms:
        condition = axiom.condition
        if len(condition) == 0:
            # The derived fact we want to negate is triggered with an
            # empty condition, so it is always true and its negation
            # is always false.
            return []
        elif len(condition) == 1:  # Handle easy special case quickly.
            new_literal = condition[0].negate()
            for result_axiom in result:
                result_axiom.condition.append(new_literal)
        else:
            new_result = []
            for literal in condition:
                literal = literal.negate()
                for result_axiom in result:
                    new_axiom = result_axiom.clone()
                    new_axiom.condition.append(literal)
                    new_result.append(new_axiom)
            result = new_result
    result = simplify(result)
    return result
