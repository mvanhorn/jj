// Copyright 2026 The Jujutsu Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::slice;
use std::sync::Arc;

use futures::StreamExt as _;
use futures::executor::block_on_stream;
use itertools::Itertools as _;
use jj_lib::backend::ChangeId;
use jj_lib::backend::CommitId;
use jj_lib::backend::MillisSinceEpoch;
use jj_lib::backend::Signature;
use jj_lib::backend::Timestamp;
use jj_lib::backend::TreeId;
use jj_lib::backend::TreeValue;
use jj_lib::commit::Commit;
use jj_lib::conflict_labels::ConflictLabels;
use jj_lib::converge::CommitsByChangeId;
use jj_lib::converge::ConvergedAttribute;
use jj_lib::converge::TreeIdsAndLabels;
use jj_lib::converge::TruncatedEvolutionGraph;
use jj_lib::converge::apply_solution;
use jj_lib::converge::converge_change;
use jj_lib::converge::find_divergent_changes;
use jj_lib::converge::remove_descendants;
use jj_lib::evolution::walk_predecessors;
use jj_lib::merge::Merge;
use jj_lib::merge::MergeBuilder;
use jj_lib::merged_tree::MergedTree;
use jj_lib::object_id::ObjectId as _;
use jj_lib::repo::ReadonlyRepo;
use jj_lib::repo::Repo;
use jj_lib::revset::RevsetExpression;
use jj_lib::store::Store;
use jj_lib::transaction::Transaction;
use pollster::FutureExt as _;
use testutils::CommitBuilderExt as _;
use testutils::TestRepo;
use testutils::TestResult;
use testutils::commit_transactions;
use testutils::create_random_tree;
use testutils::create_tree_with;
use testutils::dump_tree;
use testutils::repo_path;
use testutils::repo_path_buf;
use testutils::write_random_commit;
use testutils::write_random_commit_with_parents;

fn make_change_id(repo: &TestRepo, byte: u8) -> ChangeId {
    ChangeId::new(vec![byte; repo.repo.store().change_id_length()])
}

fn get_merged_tree_value(tree: &MergedTree, path: &str) -> TestResult<Option<TreeValue>> {
    Ok(tree
        .trees()
        .block_on()?
        .into_resolved()
        .unwrap()
        .path_value(repo_path(path))
        .block_on()?)
}

#[allow(dead_code)]
fn tree_to_string(
    store: &Arc<Store>,
    tree_ids: &Merge<TreeId>,
    conflict_labels: &ConflictLabels,
) -> String {
    dump_tree(&MergedTree::new(
        store.clone(),
        tree_ids.clone(),
        conflict_labels.clone(),
    ))
}

fn get_predecessors(repo: &ReadonlyRepo, id: &CommitId) -> Vec<CommitId> {
    let entries: Vec<_> =
        block_on_stream(walk_predecessors(repo, slice::from_ref(id)).boxed_local())
            .try_collect()
            .expect("unreachable predecessors shouldn't be visited");
    let first = entries
        .first()
        .expect("specified commit should be reachable");
    first.predecessor_ids().to_vec()
}

fn fixed_signature(name: &str, email: &str, millis: i64, tz_offset: i32) -> Signature {
    Signature {
        name: name.to_owned(),
        email: email.to_owned(),
        timestamp: Timestamp {
            timestamp: MillisSinceEpoch(millis),
            tz_offset,
        },
    }
}

fn assert_same_change_content(commits: &[Commit]) {
    let first = &commits[0];
    for commit in &commits[1..] {
        assert_eq!(commit.change_id(), first.change_id());
        assert_eq!(commit.description(), first.description());
        assert_eq!(commit.parent_ids(), first.parent_ids());
        assert_eq!(commit.tree_ids(), first.tree_ids());
    }
}

/// Rewrites `base` once per author, concurrently. Each side keeps the base
/// change id, description, parents, and tree. Committer signatures differ so
/// the rewritten commits stay distinct when their authors match.
fn fork_change_with_authors(
    test_repo: &TestRepo,
    base_author: &Signature,
    side_authors: &[Signature],
) -> TestResult<(Arc<ReadonlyRepo>, Commit, Vec<Commit>)> {
    let repo = &test_repo.repo;
    let root = repo.store().root_commit_id();
    let tree = repo.store().empty_merged_tree();
    let change_id = make_change_id(test_repo, 0xA1);

    let mut tx = repo.start_transaction();
    let base = create_commit(
        &mut tx,
        &[root],
        &tree,
        base_author,
        "description",
        Some(&change_id),
    );
    let repo_base = tx.commit("base").block_on()?;

    let mut sides = Vec::new();
    for (index, author) in side_authors.iter().enumerate() {
        let mut tx = repo_base.start_transaction();
        let commit = tx
            .repo_mut()
            .rewrite_commit(&base)
            .set_author(author.clone())
            .set_committer(fixed_signature(
                "Committer",
                "committer@example.com",
                50_000 + i64::try_from(index).unwrap(),
                0,
            ))
            .write_unwrap();
        tx.repo_mut().rebase_descendants().block_on()?;
        tx.commit("side").block_on()?;
        sides.push(commit);
    }
    let repo = repo_base.reload_at_head().block_on()?;
    assert_same_change_content(&sides);
    Ok((repo, base, sides))
}

fn converge_authors(
    repo: Arc<ReadonlyRepo>,
    commits: Vec<Commit>,
    author_override: Option<Signature>,
) -> TestResult<ConvergedAttribute<Signature>> {
    let graph = TruncatedEvolutionGraph::new(repo, commits).block_on()?;
    let result = converge_change(&graph, author_override, None, None, None).block_on()?;
    Ok(result.author)
}

fn create_commit(
    tx: &mut Transaction,
    parents: &[&CommitId],
    tree: &MergedTree,
    author: &Signature,
    desc: &str,
    change_id: Option<&ChangeId>,
) -> Commit {
    let repo = tx.repo_mut();
    let parents: Vec<CommitId> = parents.iter().map(|p| (*p).clone()).collect::<Vec<_>>();
    let builder = repo
        .new_commit(parents, tree.clone())
        .set_author(author.clone())
        .set_description(desc.to_string())
        .set_tree(tree.clone());
    match change_id {
        Some(change_id) => builder.set_change_id(change_id.clone()),
        None => builder,
    }
    .write_unwrap()
}

pub fn create_simple_tree(repo: &Arc<ReadonlyRepo>, path: &str, content: &str) -> MergedTree {
    create_tree_with(repo, |builder| {
        builder.file(&repo_path_buf(path), content);
    })
}

fn create_merged_tree(terms: Vec<(MergedTree, String)>) -> MergedTree {
    MergedTree::merge(MergeBuilder::from_iter(terms).build())
        .block_on()
        .unwrap()
}

fn assert_divergent_changes(
    repo: &Arc<ReadonlyRepo>,
    expected: &[(&ChangeId, &[Commit])],
) -> TestResult<CommitsByChangeId> {
    let expected_divergent_commits: HashMap<ChangeId, Vec<CommitId>> = expected
        .iter()
        .map(|(change_id, commits)| {
            (
                (*change_id).clone(),
                commits.iter().map(|c| c.id().clone()).collect(),
            )
        })
        .collect();
    let actual = find_divergent_changes(repo, RevsetExpression::all()).block_on()?;
    let simplified: HashMap<ChangeId, Vec<CommitId>> = actual
        .clone()
        .into_iter()
        .map(|(change_id, commits)| (change_id, commits.iter().map(|c| c.id().clone()).collect()))
        .collect();
    assert_eq!(simplified, expected_divergent_commits);
    Ok(actual)
}

fn assert_heads(repo: &dyn Repo, expected: Vec<&CommitId>) {
    let expected = expected.iter().copied().cloned().collect();
    assert_eq!(*repo.view().heads(), expected);
}

#[test]
fn test_find_divergent_changes_none_found() -> TestResult {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let root = repo.store().root_commit_id();

    let empty_tree = repo.store().empty_merged_tree();
    let author = Signature {
        name: "author1".to_string(),
        email: "author1".to_string(),
        timestamp: Timestamp::now(),
    };

    let mut tx = repo.start_transaction();
    let _commit_1 = create_commit(&mut tx, &[root], &empty_tree, &author, "commit 1", None);
    let _commit_2 = create_commit(&mut tx, &[root], &empty_tree, &author, "commit 2", None);
    let repo = tx.commit("test").block_on()?;

    let result = find_divergent_changes(&repo, RevsetExpression::all()).block_on()?;
    assert!(result.is_empty());
    Ok(())
}

#[test]
fn test_remove_descendants_linear_chain() -> TestResult {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;

    let mut tx = repo.start_transaction();
    let repo = tx.repo_mut();
    let commit1 = write_random_commit(repo);
    let commit2 = write_random_commit_with_parents(repo, &[&commit1]);
    let commit3 = write_random_commit_with_parents(repo, &[&commit2]);
    let repo = tx.commit("test").block_on()?;

    assert_eq!(
        remove_descendants(
            &repo,
            &[
                commit1.id().clone(),
                commit2.id().clone(),
                commit3.id().clone(),
            ],
        )
        .block_on()?,
        HashSet::from([commit1.id().clone()])
    );
    assert_eq!(
        remove_descendants(&repo, &[commit1.id().clone(), commit2.id().clone(),],).block_on()?,
        HashSet::from([commit1.id().clone()])
    );
    assert_eq!(
        remove_descendants(&repo, &[commit1.id().clone()],).block_on()?,
        HashSet::from([commit1.id().clone()])
    );

    Ok(())
}

#[test]
fn test_find_divergent_changes_exactly_one_found() -> TestResult {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let root = repo.store().root_commit_id();
    let change_aa = make_change_id(&test_repo, 0xAA);

    let empty_tree = repo.store().empty_merged_tree();
    let author = Signature {
        name: "author1".to_string(),
        email: "author1".to_string(),
        timestamp: Timestamp::now(),
    };

    let commit_1 = {
        let mut tx = repo.start_transaction();
        let commit = create_commit(
            &mut tx,
            &[root],
            &empty_tree,
            &author,
            "foo",
            Some(&change_aa),
        );
        tx.commit("tx1").block_on()?;
        commit
    };

    let commit_2 = {
        let mut tx = repo.start_transaction();
        let commit = create_commit(
            &mut tx,
            &[root],
            &empty_tree,
            &author,
            "bar",
            Some(&change_aa),
        );
        tx.commit("tx2").block_on()?;
        commit
    };

    let repo = repo.reload_at_head().block_on()?;
    assert_eq!(
        find_divergent_changes(&repo, RevsetExpression::all()).block_on()?,
        BTreeMap::from([(change_aa.clone(), vec![commit_2.clone(), commit_1.clone()])])
    );

    Ok(())
}

#[test]
fn test_find_divergent_changes_two_found() -> TestResult {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let root = repo.store().root_commit_id();
    let change_aa = make_change_id(&test_repo, 0xAA);
    let change_bb = make_change_id(&test_repo, 0xBB);

    let empty_tree = repo.store().empty_merged_tree();
    let author = Signature {
        name: "author1".to_string(),
        email: "author1".to_string(),
        timestamp: Timestamp::now(),
    };

    let commit_1 = {
        let mut tx = repo.start_transaction();
        let commit = create_commit(
            &mut tx,
            &[root],
            &empty_tree,
            &author,
            "foo",
            Some(&change_aa),
        );
        tx.commit("tx1").block_on()?;
        commit
    };

    let commit_2 = {
        let mut tx = repo.start_transaction();
        let commit = create_commit(
            &mut tx,
            &[root],
            &empty_tree,
            &author,
            "bar",
            Some(&change_aa),
        );
        tx.commit("tx2").block_on()?;
        commit
    };

    let commit_3 = {
        let mut tx = repo.start_transaction();
        let commit = create_commit(
            &mut tx,
            &[root],
            &empty_tree,
            &author,
            "baz",
            Some(&change_bb),
        );
        tx.commit("tx3").block_on()?;
        commit
    };

    let commit_4 = {
        let mut tx = repo.start_transaction();
        let commit = create_commit(
            &mut tx,
            &[root],
            &empty_tree,
            &author,
            "qux",
            Some(&change_bb),
        );
        tx.commit("tx4").block_on()?;
        commit
    };

    let repo = repo.reload_at_head().block_on()?;
    drop(assert_divergent_changes(
        &repo,
        &[
            (&change_aa, &[commit_2.clone(), commit_1.clone()]),
            (&change_bb, &[commit_4.clone(), commit_3.clone()]),
        ],
    )?);
    Ok(())
}

#[test]
fn test_build_truncated_evolution_graph() -> TestResult {
    let test_repo = TestRepo::init();

    let mut tx = test_repo.repo.start_transaction();
    let commit1 = write_random_commit(tx.repo_mut());
    let repo1 = tx.commit("tx1").block_on()?;

    let commit2 = {
        let mut tx = repo1.start_transaction();
        let commit2 = tx
            .repo_mut()
            .rewrite_commit(&commit1)
            .set_description("rewritten->foo")
            .write_unwrap();
        tx.repo_mut().rebase_descendants().block_on()?;
        tx.commit("tx2").block_on()?;
        commit2
    };

    let commit3 = {
        let mut tx = repo1.start_transaction();
        let commit3 = tx
            .repo_mut()
            .rewrite_commit(&commit1)
            .set_description("rewritten->bar")
            .write_unwrap();
        tx.repo_mut().rebase_descendants().block_on()?;
        tx.commit("tx3").block_on()?;
        commit3
    };

    let repo = repo1.reload_at_head().block_on()?;

    let divergent_commits = vec![commit2.clone(), commit3.clone()];
    let truncated_evolution_graph =
        TruncatedEvolutionGraph::new(repo, divergent_commits).block_on()?;
    assert_eq!(truncated_evolution_graph.change_id(), commit1.change_id());
    assert_eq!(
        truncated_evolution_graph.divergent_commit_ids(),
        &[commit2.id().clone(), commit3.id().clone()]
    );
    assert_eq!(
        truncated_evolution_graph
            .flow_graph
            .graph
            .adjacent_nodes(commit1.id())
            .unwrap()
            .collect::<Vec<_>>(),
        &[commit2.id(), commit3.id()]
    );
    assert!(
        truncated_evolution_graph
            .flow_graph
            .graph
            .adjacent_nodes(commit2.id())
            .unwrap()
            .collect::<Vec<_>>()
            .is_empty(),
    );
    assert!(
        truncated_evolution_graph
            .flow_graph
            .graph
            .adjacent_nodes(commit3.id())
            .unwrap()
            .collect::<Vec<_>>()
            .is_empty(),
    );

    Ok(())
}

#[test]
fn test_simple_converge_description() -> TestResult {
    let test_repo = TestRepo::init();

    let mut tx = test_repo.repo.start_transaction();
    let commit1 = write_random_commit(tx.repo_mut());
    let repo1 = tx.commit("tx1").block_on()?;

    let commit2 = {
        let mut tx = repo1.start_transaction();
        let commit2 = tx
            .repo_mut()
            .rewrite_commit(&commit1)
            .set_description("rewritten->foo")
            .write_unwrap();
        tx.repo_mut().rebase_descendants().block_on()?;
        tx.commit("tx2").block_on()?;
        commit2
    };

    let commit3 = {
        let mut tx = repo1.start_transaction();
        let commit3 = tx
            .repo_mut()
            .rewrite_commit(&commit1)
            .set_description("rewritten->bar")
            .write_unwrap();
        tx.repo_mut().rebase_descendants().block_on()?;
        tx.commit("tx3").block_on()?;
        commit3
    };

    let repo = repo1.reload_at_head().block_on()?;
    let divergent_commits = vec![commit2.clone(), commit3.clone()];
    let truncated_evolution_graph =
        TruncatedEvolutionGraph::new(repo, divergent_commits).block_on()?;

    let converge_result =
        converge_change(&truncated_evolution_graph, None, None, None, None).block_on()?;
    assert_eq!(
        converge_result.description,
        ConvergedAttribute::Unsolved {
            base_commit: commit1.id().clone(),
            excluded_divergent_commits: HashSet::default()
        }
    );

    let converge_result = converge_change(
        &truncated_evolution_graph,
        None,
        Some("user-merged description".to_string()),
        None,
        None,
    )
    .block_on()?;
    assert_eq!(
        converge_result.description,
        ConvergedAttribute::Solved("user-merged description".to_string())
    );
    Ok(())
}

// Evolution (predecessors are below their successors):
//
// C4  C5
// |   |
// C2  C3
//  \  /
//   C1
//
// C1 is rewritten to C2 and C3 in parallel, and then in a single transaction C2
// is rewritten to C4 and C3 is rewritten to C5. The visible commits at the end
// are C4 and C5. The only thing changing throughout is the description.
#[test]
fn test_manual_converge_description_concurrent_ops() -> TestResult {
    let test_repo = TestRepo::init();
    let repo0 = test_repo.repo;

    let mut tx = repo0.start_transaction();
    let commit1 = write_random_commit(tx.repo_mut());
    let repo1 = tx.commit("test").block_on()?;

    let mut tx2 = repo1.start_transaction();
    let commit2 = tx2
        .repo_mut()
        .rewrite_commit(&commit1)
        .set_description("rewritten 2")
        .write_unwrap();
    tx2.repo_mut().rebase_descendants().block_on()?;
    let mut tx3 = repo1.start_transaction();
    let commit3 = tx3
        .repo_mut()
        .rewrite_commit(&commit1)
        .set_description("rewritten 3")
        .write_unwrap();
    tx3.repo_mut().rebase_descendants().block_on()?;
    let repo4 = commit_transactions(vec![tx2, tx3]);

    let mut tx = repo4.start_transaction();
    let commit4 = tx
        .repo_mut()
        .rewrite_commit(&commit2)
        .set_description("rewritten 4")
        .write_unwrap();
    let commit5 = tx
        .repo_mut()
        .rewrite_commit(&commit3)
        .set_description("rewritten 5")
        .write_unwrap();
    tx.repo_mut().rebase_descendants().block_on()?;
    let repo5 = tx.commit("test").block_on()?;

    let change_id = commit1.change_id().clone();
    assert_eq!(
        find_divergent_changes(&repo5, RevsetExpression::all()).block_on()?,
        BTreeMap::from([(change_id.clone(), vec![commit5.clone(), commit4.clone()])])
    );

    let divergent_commits = vec![commit4.clone(), commit5.clone()];
    let truncated_evolution_graph =
        TruncatedEvolutionGraph::new(repo5, divergent_commits).block_on()?;

    let converge_result =
        converge_change(&truncated_evolution_graph, None, None, None, None).block_on()?;
    assert_eq!(
        converge_result.description,
        ConvergedAttribute::Unsolved {
            base_commit: commit1.id().clone(),
            excluded_divergent_commits: HashSet::default()
        }
    );

    let converge_result = converge_change(
        &truncated_evolution_graph,
        None,
        Some("user-merged description".to_string()),
        None,
        None,
    )
    .block_on()?;
    assert_eq!(
        converge_result.description,
        ConvergedAttribute::Solved("user-merged description".to_string())
    );
    assert_eq!(
        converge_result.author,
        ConvergedAttribute::Solved(commit1.author().clone())
    );
    assert_eq!(
        converge_result.parents,
        ConvergedAttribute::Solved(commit1.parent_ids().to_vec())
    );
    assert_eq!(
        converge_result.tree,
        Some(TreeIdsAndLabels::new(commit1.tree()))
    );
    Ok(())
}

// Evolution (predecessors are below their successors):
//
// C4("baz", parent_x)
//      |
// C2("bar", parent_y)      C3("bar", parent_x)
//      \                      /
//       C1("foo", parent_x)
//
// C1 is rewritten to C2 and C3 in parallel, and then C2 is rewritten to C4. The
// visible commits at the end are C3 and C4. converge is possible without user
// input.
//
// Expected result: Solution("baz", parent_x).
#[test]
fn test_automatic_converge_description_and_parent() -> TestResult {
    let test_repo = TestRepo::init();

    // First create the parents.
    let mut tx = test_repo.repo.start_transaction();
    let parent_x = write_random_commit(tx.repo_mut()).id().clone();
    let parent_y = write_random_commit(tx.repo_mut()).id().clone();
    let repo0 = tx.commit("test").block_on()?;

    let mut tx = repo0.start_transaction();
    let tree = create_random_tree(tx.repo_mut().base_repo());
    let commit1 = tx
        .repo_mut()
        .new_commit(vec![parent_x.clone()], tree)
        .set_description("foo".to_string())
        .write_unwrap();
    let repo1 = tx.commit("test").block_on()?;

    let mut tx2 = repo1.start_transaction();
    let commit2 = tx2
        .repo_mut()
        .rewrite_commit(&commit1)
        .set_description("bar")
        .set_parents(vec![parent_y.clone()])
        .write_unwrap();
    tx2.repo_mut().rebase_descendants().block_on()?;
    let mut tx3 = repo1.start_transaction();
    let commit3 = tx3
        .repo_mut()
        .rewrite_commit(&commit1)
        .set_description("bar")
        .write_unwrap();
    tx3.repo_mut().rebase_descendants().block_on()?;
    let repo4 = commit_transactions(vec![tx2, tx3]);

    let mut tx = repo4.start_transaction();
    let commit4 = tx
        .repo_mut()
        .rewrite_commit(&commit2)
        .set_description("baz")
        .set_parents(vec![parent_x.clone()])
        .write_unwrap();
    tx.repo_mut().rebase_descendants().block_on()?;
    let repo5 = tx.commit("test").block_on()?;

    let change_id = commit1.change_id().clone();
    let divergent_commits = vec![commit4.clone(), commit3.clone()];
    assert_divergent_changes(&repo5, &[(&change_id, &divergent_commits)])?;

    let truncated_evolution_graph =
        TruncatedEvolutionGraph::new(repo5, divergent_commits).block_on()?;
    let converge_result =
        converge_change(&truncated_evolution_graph, None, None, None, None).block_on()?;
    assert_eq!(
        converge_result.description,
        ConvergedAttribute::Solved("baz".to_string())
    );
    assert_eq!(
        converge_result.author,
        ConvergedAttribute::Solved(commit1.author().clone())
    );
    assert_eq!(
        converge_result.parents,
        ConvergedAttribute::Solved(vec![parent_x.clone()])
    );
    assert_eq!(
        converge_result.tree,
        Some(TreeIdsAndLabels::new(commit1.tree()))
    );
    Ok(())
}

// Evolution (predecessors are below their successors):
//
// C4("baz", parent:X, file="content4")
//      |
// C2("bar", parent:Y, file="content2")
//      |
//      |                           C3("bar", parent:X,file="content3")
//       \                                    /
//       C1("foo", parent:X, file="content1")
//
// C1 is rewritten to C2 and C3 in parallel, and then C2 is rewritten to C4. The
// visible commits at the end are base,X,Y,C3,C4. converge is possible without
// user input.
//
// Commit graph:
//
// C1 C3 C4  C2
//  \ | /    |
//    X      Y
//     \    /
//      base
//
// Expected result (commit graph):
//
// Solution("baz")
//    |
//    X       Y
//     \     /
//      base
#[test]
fn test_automatic_converge_description_parent_and_trees() -> TestResult {
    let test_repo = TestRepo::init();
    let root = test_repo.repo.store().root_commit_id();
    let change_aa = make_change_id(&test_repo, 0xAA);
    let change_bb = make_change_id(&test_repo, 0xBB);
    let change_cc = make_change_id(&test_repo, 0xCC);

    let tree_base = create_simple_tree(&test_repo.repo, "otherfile", "content: otherfile");
    let tree_x = create_simple_tree(&test_repo.repo, "file", "content: X");
    let tree_y = create_simple_tree(&test_repo.repo, "file", "content: Y");
    let tree1 = create_simple_tree(&test_repo.repo, "file", "content1");
    let tree2 = create_simple_tree(&test_repo.repo, "file", "content2");
    let tree3 = create_simple_tree(&test_repo.repo, "file", "content3");
    let tree4 = create_simple_tree(&test_repo.repo, "file", "content4");

    // First create the parents.
    let mut tx = test_repo.repo.start_transaction();
    let base = tx
        .repo_mut()
        .new_commit(vec![root.clone()], tree_base.clone())
        .set_description("base".to_string())
        .write_unwrap()
        .id()
        .clone();
    let commit_x = tx
        .repo_mut()
        .new_commit(vec![base.clone()], tree_x.clone())
        .set_change_id(change_aa)
        .set_description("X".to_string())
        .write_unwrap();
    let commit_y = tx
        .repo_mut()
        .new_commit(vec![base.clone()], tree_y)
        .set_change_id(change_bb)
        .set_description("Y".to_string())
        .write_unwrap();
    let repo0 = tx.commit("test").block_on()?;

    let mut tx = repo0.start_transaction();
    let commit1 = tx
        .repo_mut()
        .new_commit(vec![commit_x.id().clone()], tree1.clone())
        .set_change_id(change_cc.clone())
        .set_description("foo".to_string())
        .write_unwrap();
    let repo1 = tx.commit("test").block_on()?;

    let mut tx2 = repo1.start_transaction();
    let commit2 = tx2
        .repo_mut()
        .rewrite_commit(&commit1)
        .set_description("bar")
        .set_parents(vec![commit_y.id().clone()])
        .set_tree(tree2)
        .write_unwrap();
    tx2.repo_mut().rebase_descendants().block_on()?;
    let mut tx3 = repo1.start_transaction();
    let commit3 = tx3
        .repo_mut()
        .rewrite_commit(&commit1)
        .set_description("bar")
        .set_tree(tree3.clone())
        .write_unwrap();
    tx3.repo_mut().rebase_descendants().block_on()?;
    let repo4 = commit_transactions(vec![tx2, tx3]);

    let mut tx = repo4.start_transaction();
    let commit4 = tx
        .repo_mut()
        .rewrite_commit(&commit2)
        .set_description("baz")
        .set_parents(vec![commit_x.id().clone()])
        .set_tree(tree4.clone())
        .write_unwrap();
    tx.repo_mut().rebase_descendants().block_on()?;
    let repo5 = tx.commit("test").block_on()?;

    let change_id = commit1.change_id().clone();
    let divergent_commits = vec![commit4.clone(), commit3.clone()];
    let divergent_commit_ids = divergent_commits
        .iter()
        .map(|c| c.id().clone())
        .collect_vec();
    assert_divergent_changes(&repo5, &[(&change_id, &divergent_commits)])?;

    let truncated_evolution_graph =
        TruncatedEvolutionGraph::new(repo5.clone(), divergent_commits.clone()).block_on()?;
    let converge_result =
        converge_change(&truncated_evolution_graph, None, None, None, None).block_on()?;

    let expected_tree = create_merged_tree(vec![
        (
            commit4.tree().clone(),
            format!("divergent commit: {}", commit4.conflict_label()),
        ),
        (
            commit1.tree().clone(),
            format!("converge base: {}", commit1.conflict_label()),
        ),
        (
            commit3.tree().clone(),
            format!("divergent commit: {}", commit3.conflict_label()),
        ),
        (
            commit1.tree().clone(),
            format!("converge base: {}", commit1.conflict_label()),
        ),
        (
            commit1.tree().clone(),
            format!("converge base: {}", commit1.conflict_label()),
        ),
    ]);

    assert_eq!(
        converge_result.description,
        ConvergedAttribute::Solved("baz".to_string())
    );
    assert_eq!(
        converge_result.author,
        ConvergedAttribute::Solved(commit1.author().clone())
    );
    assert_eq!(
        converge_result.parents,
        ConvergedAttribute::Solved(vec![commit_x.id().clone()])
    );
    assert_eq!(
        converge_result.tree,
        Some(TreeIdsAndLabels::new(expected_tree.clone()))
    );

    // TODO
    // assert_eq!(
    //     tree_to_string(
    //         test_repo.repo.store(),
    //         &solution.tree_ids,
    //         &solution.conflict_labels
    //     ),
    //     "xyz"
    // );

    let mut tx = repo5.start_transaction();
    let (applied, _) = apply_solution(
        commit1.author().clone(),
        "baz".to_string(),
        vec![commit_x.id().clone()],
        TreeIdsAndLabels::new(expected_tree),
        change_id,
        &divergent_commit_ids,
        tx.repo_mut(),
    )
    .block_on()?;
    let repo = tx.commit("apply solution").block_on()?;
    assert_heads(repo.as_ref(), vec![applied.id(), commit_y.id()]);

    assert_eq!(applied.change_id(), &change_cc);
    assert_eq!(applied.description(), "baz");
    assert_eq!(applied.parent_ids(), &[commit_x.id().clone()]);

    assert_eq!(
        applied
            .tree()
            .path_value(repo_path("file"))
            .block_on()
            .unwrap(),
        Merge::from_removes_adds(
            vec![get_merged_tree_value(&tree1, "file")?],
            vec![
                get_merged_tree_value(&tree4, "file")?,
                get_merged_tree_value(&tree3, "file")?,
            ],
        ),
    );
    assert_eq!(get_predecessors(&repo, applied.id()), divergent_commit_ids);
    Ok(())
}

// Evolution (predecessors are below their successors):
//
// C4("baz", parent:Y, file="content4")
//      |
// C2("bar", parent:Y, file="content2")
//      |
//      |                           C3("bar", parent:X,file="content3")
//       \                                    /
//       C1("foo", parent:X, file="content1")
//
// C1 is rewritten to C2 and C3 in parallel, and then C2 is rewritten to C4. The
// visible commits at the end are base,X,Y,C3,C4. converge is possible without
// user input.
//
// Commit graph:
//
// C1 C3  C2 C4
//  \ /    \ /
//   X      Y
//    \    /
//     base
//
// Expected result (commit graph):
//
//         Solution("baz")
//            |
//    X       Y
//     \     /
//      base
#[test]
fn test_automatic_converge_description_parent_and_trees_with_reparent() -> TestResult {
    let test_repo = TestRepo::init();
    let root = test_repo.repo.store().root_commit_id();
    let change_aa = make_change_id(&test_repo, 0xAA);
    let change_bb = make_change_id(&test_repo, 0xBB);
    let change_cc = make_change_id(&test_repo, 0xCC);

    let tree_base = create_simple_tree(&test_repo.repo, "otherfile", "content: otherfile");
    let tree_x = create_simple_tree(&test_repo.repo, "file", "content: X");
    let tree_y = create_simple_tree(&test_repo.repo, "file", "content: Y");
    let tree1 = create_simple_tree(&test_repo.repo, "file", "content1");
    let tree2 = create_simple_tree(&test_repo.repo, "file", "content2");
    let tree3 = create_simple_tree(&test_repo.repo, "file", "content3");
    let tree4 = create_simple_tree(&test_repo.repo, "file", "content4");

    // First create the parents.
    let mut tx = test_repo.repo.start_transaction();
    let base = tx
        .repo_mut()
        .new_commit(vec![root.clone()], tree_base.clone())
        .set_description("base".to_string())
        .write_unwrap()
        .id()
        .clone();
    let commit_x = tx
        .repo_mut()
        .new_commit(vec![base.clone()], tree_x.clone())
        .set_change_id(change_aa)
        .set_description("X".to_string())
        .write_unwrap();
    let commit_y = tx
        .repo_mut()
        .new_commit(vec![base.clone()], tree_y)
        .set_change_id(change_bb)
        .set_description("Y".to_string())
        .write_unwrap();
    let repo0 = tx.commit("test").block_on()?;

    let mut tx = repo0.start_transaction();
    let commit1 = tx
        .repo_mut()
        .new_commit(vec![commit_x.id().clone()], tree1.clone())
        .set_change_id(change_cc.clone())
        .set_description("foo".to_string())
        .write_unwrap();
    let repo1 = tx.commit("test").block_on()?;

    let mut tx2 = repo1.start_transaction();
    let commit2 = tx2
        .repo_mut()
        .rewrite_commit(&commit1)
        .set_description("bar")
        .set_parents(vec![commit_y.id().clone()])
        .set_tree(tree2)
        .write_unwrap();
    tx2.repo_mut().rebase_descendants().block_on()?;
    let mut tx3 = repo1.start_transaction();
    let commit3 = tx3
        .repo_mut()
        .rewrite_commit(&commit1)
        .set_description("bar")
        .set_parents(vec![commit_x.id().clone()])
        .set_tree(tree3.clone())
        .write_unwrap();
    tx3.repo_mut().rebase_descendants().block_on()?;
    let repo4 = commit_transactions(vec![tx2, tx3]);

    let mut tx = repo4.start_transaction();
    let commit4 = tx
        .repo_mut()
        .rewrite_commit(&commit2)
        .set_description("baz")
        .set_parents(vec![commit_y.id().clone()])
        .set_tree(tree4.clone())
        .write_unwrap();
    tx.repo_mut().rebase_descendants().block_on()?;
    let repo5 = tx.commit("test").block_on()?;

    let change_id = commit1.change_id().clone();
    let divergent_commits = vec![commit4.clone(), commit3.clone()];
    assert_divergent_changes(&repo5, &[(&change_id, &divergent_commits)])?;
    let divergent_commit_ids = divergent_commits
        .iter()
        .map(|c| c.id().clone())
        .collect_vec();
    let truncated_evolution_graph =
        TruncatedEvolutionGraph::new(repo5.clone(), divergent_commits.clone()).block_on()?;
    let converge_result =
        converge_change(&truncated_evolution_graph, None, None, None, None).block_on()?;

    let rebased_tree1 = create_merged_tree(vec![
        (
            commit_y.tree().clone(),
            "converge solution parent(s)".to_string(),
        ),
        (
            commit_x.tree().clone(),
            format!(
                "(negated) nnnnnnnn {} \"{}\"",
                &commit_x.id().hex()[0..8],
                commit_x.description()
            ),
        ),
        (
            commit1.tree().clone(),
            format!(
                "nnnnnnnn {} \"{}\"",
                &commit1.id().hex()[0..8],
                commit1.description()
            ),
        ),
    ]);
    let rebased_tree3 = create_merged_tree(vec![
        (
            commit_y.tree().clone(),
            "converge solution parent(s)".to_string(),
        ),
        (
            commit_x.tree().clone(),
            format!(
                "(negated) nnnnnnnn {} \"{}\"",
                &commit_x.id().hex()[0..8],
                commit_x.description()
            ),
        ),
        (
            commit3.tree().clone(),
            format!(
                "nnnnnnnn {} \"{}\"",
                &commit3.id().hex()[0..8],
                commit3.description()
            ),
        ),
    ]);
    let rebased_tree4 = tree4.clone();

    let expected_tree = create_merged_tree(vec![
        (
            rebased_tree1.clone(),
            format!(
                "converge base: nnnnnnnn {} \"{}\"",
                &commit1.id().hex()[0..8],
                commit1.description()
            ),
        ),
        (
            rebased_tree1.clone(),
            format!(
                "converge base: nnnnnnnn {} \"{}\"",
                &commit1.id().hex()[0..8],
                commit1.description()
            ),
        ),
        (
            rebased_tree3.clone(),
            format!(
                "divergent commit: nnnnnnnn {} \"{}\"",
                &commit3.id().hex()[0..8],
                commit3.description()
            ),
        ),
        (
            rebased_tree1.clone(),
            format!(
                "converge base: nnnnnnnn {} \"{}\"",
                &commit1.id().hex()[0..8],
                commit1.description()
            ),
        ),
        (
            rebased_tree4.clone(),
            format!(
                "divergent commit: nnnnnnnn {} \"{}\"",
                &commit4.id().hex()[0..8],
                commit4.description()
            ),
        ),
    ]);

    assert_eq!(
        converge_result.description,
        ConvergedAttribute::Solved("baz".to_string())
    );
    assert_eq!(
        converge_result.author,
        ConvergedAttribute::Solved(commit1.author().clone())
    );
    assert_eq!(
        converge_result.parents,
        ConvergedAttribute::Solved(vec![commit_y.id().clone()])
    );
    assert_eq!(
        converge_result.tree,
        Some(TreeIdsAndLabels::new(expected_tree.clone()))
    );

    // TODO
    // assert_eq!(
    //     tree_to_string(
    //         test_repo.repo.store(),
    //         &solution.tree_ids,
    //         &solution.conflict_labels
    //     ),
    //     "xyz"
    // );

    let mut tx = repo5.start_transaction();
    let (applied, _) = apply_solution(
        commit1.author().clone(),
        "baz".to_string(),
        vec![commit_y.id().clone()],
        TreeIdsAndLabels::new(expected_tree),
        change_id,
        &divergent_commit_ids,
        tx.repo_mut(),
    )
    .block_on()?;
    let repo = tx.commit("apply solution").block_on()?;

    assert_heads(repo.as_ref(), vec![applied.id(), commit_x.id()]);
    assert_eq!(applied.change_id(), &change_cc);
    assert_eq!(applied.description(), "baz");
    assert_eq!(applied.parent_ids(), &[commit_y.id().clone()]);
    assert_eq!(
        applied.tree().path_value(repo_path("file")).block_on()?,
        Merge::from_removes_adds(
            vec![get_merged_tree_value(&tree1, "file")?],
            vec![
                get_merged_tree_value(&tree4, "file")?,
                get_merged_tree_value(&tree3, "file")?,
            ],
        ),
    );
    assert_eq!(get_predecessors(&repo, applied.id()), divergent_commit_ids);
    Ok(())
}

#[test]
fn test_converge_author_identical_signatures() -> TestResult {
    let test_repo = TestRepo::init();
    let author = fixed_signature("Alice", "alice@example.com", 1_000, 0);
    let (repo, _base, sides) =
        fork_change_with_authors(&test_repo, &author, &[author.clone(), author.clone()])?;

    assert_eq!(sides[0].author(), sides[1].author());
    assert_eq!(sides[0].author().timestamp, author.timestamp);

    let solved = converge_authors(repo, sides, None)?;
    assert_eq!(solved, ConvergedAttribute::Solved(author));
    Ok(())
}

#[test]
fn test_converge_author_same_identity_different_timestamps() -> TestResult {
    let test_repo = TestRepo::init();
    let base = fixed_signature("Alice", "alice@example.com", 1_000, 0);
    let left = fixed_signature("Alice", "alice@example.com", 2_000, 0);
    let right = fixed_signature("Alice", "alice@example.com", 3_500, 60);
    let (repo, _base_commit, sides) =
        fork_change_with_authors(&test_repo, &base, &[left.clone(), right.clone()])?;

    assert_eq!(sides[0].description(), sides[1].description());
    assert_eq!(sides[0].parent_ids(), sides[1].parent_ids());
    assert_eq!(sides[0].tree_ids(), sides[1].tree_ids());
    assert_eq!(sides[0].change_id(), sides[1].change_id());
    assert_eq!(sides[0].author().name, sides[1].author().name);
    assert_eq!(sides[0].author().email, sides[1].author().email);
    assert_ne!(sides[0].author().timestamp, sides[1].author().timestamp);
    assert_eq!(sides[0].author(), &left);
    assert_eq!(sides[1].author(), &right);

    let graph = TruncatedEvolutionGraph::new(repo.clone(), sides.clone()).block_on()?;
    let result = converge_change(&graph, None, None, None, None).block_on()?;
    assert_eq!(result.author, ConvergedAttribute::Solved(left.clone()));
    assert_eq!(
        result.description,
        ConvergedAttribute::Solved("description".to_string())
    );
    assert_eq!(
        result.parents,
        ConvergedAttribute::Solved(sides[0].parent_ids().to_vec())
    );

    // Reversing the graph's input order keeps the identity and selects the other
    // original timestamp.
    let reversed = vec![sides[1].clone(), sides[0].clone()];
    let solved = converge_authors(repo, reversed, None)?;
    assert_eq!(solved, ConvergedAttribute::Solved(right));
    Ok(())
}

#[test]
fn test_converge_author_three_matching_identities_timestamp_tie_break() -> TestResult {
    let test_repo = TestRepo::init();
    let base = fixed_signature("Alice", "alice@example.com", 1_000, 0);
    let first = fixed_signature("Alice", "alice@example.com", 1_600_000_000_000, -480);
    let second = fixed_signature("Alice", "alice@example.com", 1_700_000_000_000, 0);
    let third = fixed_signature("Alice", "alice@example.com", 1_800_000_000_000, 540);
    let (repo, _base_commit, sides) = fork_change_with_authors(
        &test_repo,
        &base,
        &[first.clone(), second.clone(), third.clone()],
    )?;

    assert_eq!(sides[0].author(), &first);
    assert_eq!(sides[1].author(), &second);
    assert_eq!(sides[2].author(), &third);
    assert_ne!(first.timestamp, second.timestamp);
    assert_ne!(second.timestamp, third.timestamp);
    assert_ne!(first.timestamp.tz_offset, third.timestamp.tz_offset);

    let solved = converge_authors(repo.clone(), sides.clone(), None)?;
    assert_eq!(solved, ConvergedAttribute::Solved(first));

    let mut reversed = sides;
    reversed.reverse();
    let solved = converge_authors(repo, reversed, None)?;
    assert_eq!(solved, ConvergedAttribute::Solved(third));
    Ok(())
}

#[test]
fn test_converge_author_one_side_changes_identity_other_side_timestamp() -> TestResult {
    let test_repo = TestRepo::init();
    let author_a = fixed_signature("Alice", "alice@example.com", 1_000, 0);
    let author_a_later = fixed_signature("Alice", "alice@example.com", 2_000, -120);
    let author_b = fixed_signature("Bob", "bob@example.com", 3_000, 60);
    let (repo, base, sides) = fork_change_with_authors(
        &test_repo,
        &author_a,
        &[author_a_later.clone(), author_b.clone()],
    )?;

    assert_eq!(base.author(), &author_a);
    assert_eq!(sides[0].author(), &author_a_later);
    assert_eq!(sides[0].author().name, base.author().name);
    assert_eq!(sides[0].author().email, base.author().email);
    assert_ne!(sides[0].author().timestamp, base.author().timestamp);
    assert_eq!(sides[1].author(), &author_b);

    // The timestamp-only commit is first. Identity still resolves to Bob, and the
    // signature is Bob's original one.
    let solved = converge_authors(repo.clone(), sides.clone(), None)?;
    assert_eq!(solved, ConvergedAttribute::Solved(author_b.clone()));

    let reversed = vec![sides[1].clone(), sides[0].clone()];
    let solved = converge_authors(repo, reversed, None)?;
    assert_eq!(solved, ConvergedAttribute::Solved(author_b));
    Ok(())
}

#[test]
fn test_converge_author_both_sides_same_new_identity_different_timestamps() -> TestResult {
    let test_repo = TestRepo::init();
    let author_a = fixed_signature("Alice", "alice@example.com", 1_000, 0);
    let author_b1 = fixed_signature("Bob", "bob@example.com", 4_000, 0);
    let author_b2 = fixed_signature("Bob", "bob@example.com", 5_000, 180);
    let (repo, base, sides) = fork_change_with_authors(
        &test_repo,
        &author_a,
        &[author_b1.clone(), author_b2.clone()],
    )?;

    assert_eq!(base.author().name, "Alice");
    assert_eq!(sides[0].author().name, sides[1].author().name);
    assert_eq!(sides[0].author().email, sides[1].author().email);
    assert_ne!(sides[0].author().timestamp, sides[1].author().timestamp);
    assert_ne!(sides[0].author().timestamp, base.author().timestamp);

    let solved = converge_authors(repo.clone(), sides.clone(), None)?;
    assert_eq!(solved, ConvergedAttribute::Solved(author_b1));

    let reversed = vec![sides[1].clone(), sides[0].clone()];
    let solved = converge_authors(repo, reversed, None)?;
    assert_eq!(solved, ConvergedAttribute::Solved(author_b2));
    Ok(())
}

// Evolution (predecessors are below their successors):
//
// Carol ts4000          Bob ts3000 tz+120
//     |                     |
// Bob ts2000                |
//      \                   /
//       Alice ts1000
//
// The hidden Bob and the visible Bob are the same identity with different
// timestamps. Value-flow has to treat them as one value so Carol wins. If
// timestamps split those Bobs, the dominator falls back to Alice and the
// author merge stays unsolved.
#[test]
fn test_converge_author_historical_identity_ignores_timestamps() -> TestResult {
    let test_repo = TestRepo::init();
    let repo = &test_repo.repo;
    let root = repo.store().root_commit_id();
    let tree = repo.store().empty_merged_tree();
    let change_id = make_change_id(&test_repo, 0xB2);

    let author_a = fixed_signature("Alice", "alice@example.com", 1_000, 0);
    let author_b_hist = fixed_signature("Bob", "bob@example.com", 2_000, 0);
    let author_b_visible = fixed_signature("Bob", "bob@example.com", 3_000, 120);
    let author_c = fixed_signature("Carol", "carol@example.com", 4_000, -60);

    let mut tx = repo.start_transaction();
    let commit_a = create_commit(
        &mut tx,
        &[root],
        &tree,
        &author_a,
        "description",
        Some(&change_id),
    );
    let repo_a = tx.commit("base").block_on()?;

    let mut tx = repo_a.start_transaction();
    let commit_b_hist = tx
        .repo_mut()
        .rewrite_commit(&commit_a)
        .set_author(author_b_hist.clone())
        .set_committer(fixed_signature(
            "Committer",
            "committer@example.com",
            60_000,
            0,
        ))
        .write_unwrap();
    tx.repo_mut().rebase_descendants().block_on()?;
    let repo_b = tx.commit("to bob").block_on()?;

    let mut tx = repo_b.start_transaction();
    let commit_c = tx
        .repo_mut()
        .rewrite_commit(&commit_b_hist)
        .set_author(author_c.clone())
        .set_committer(fixed_signature(
            "Committer",
            "committer@example.com",
            60_001,
            0,
        ))
        .write_unwrap();
    tx.repo_mut().rebase_descendants().block_on()?;
    tx.commit("to carol").block_on()?;

    let mut tx = repo_a.start_transaction();
    let commit_b = tx
        .repo_mut()
        .rewrite_commit(&commit_a)
        .set_author(author_b_visible.clone())
        .set_committer(fixed_signature(
            "Committer",
            "committer@example.com",
            60_002,
            0,
        ))
        .write_unwrap();
    tx.repo_mut().rebase_descendants().block_on()?;
    tx.commit("other bob").block_on()?;

    let repo = repo_a.reload_at_head().block_on()?;
    assert_eq!(
        get_predecessors(&repo, commit_c.id()),
        vec![commit_b_hist.id().clone()]
    );
    assert_eq!(
        get_predecessors(&repo, commit_b_hist.id()),
        vec![commit_a.id().clone()]
    );
    assert_eq!(
        get_predecessors(&repo, commit_b.id()),
        vec![commit_a.id().clone()]
    );
    assert_eq!(commit_b_hist.author().name, commit_b.author().name);
    assert_eq!(commit_b_hist.author().email, commit_b.author().email);
    assert_ne!(
        commit_b_hist.author().timestamp,
        commit_b.author().timestamp
    );
    assert_same_change_content(&[commit_c.clone(), commit_b.clone()]);

    let solved = converge_authors(repo.clone(), vec![commit_b.clone(), commit_c.clone()], None)?;
    assert_eq!(solved, ConvergedAttribute::Solved(author_c.clone()));

    let solved = converge_authors(repo, vec![commit_c, commit_b], None)?;
    assert_eq!(solved, ConvergedAttribute::Solved(author_c));
    Ok(())
}

#[test]
fn test_converge_author_conflicting_identities_stay_unsolved() -> TestResult {
    let author_a = fixed_signature("Alice", "alice@example.com", 1_000, 0);
    let author_b = fixed_signature("Bob", "bob@example.com", 1_000, 0);
    let author_c = fixed_signature("Carol", "carol@example.com", 1_000, 0);

    let test_repo = TestRepo::init();
    let (repo, base, sides) =
        fork_change_with_authors(&test_repo, &author_a, &[author_b.clone(), author_c.clone()])?;
    assert_eq!(sides[0].author().timestamp, sides[1].author().timestamp);
    let solved = converge_authors(repo, sides, None)?;
    assert_eq!(
        solved,
        ConvergedAttribute::Unsolved {
            base_commit: base.id().clone(),
            excluded_divergent_commits: HashSet::default(),
        }
    );

    // Same email and timestamp, different names.
    let test_repo = TestRepo::init();
    let name_b = fixed_signature("Bob", "alice@example.com", 1_000, 0);
    let name_c = fixed_signature("Carol", "alice@example.com", 1_000, 0);
    let (repo, base, sides) = fork_change_with_authors(&test_repo, &author_a, &[name_b, name_c])?;
    assert_eq!(sides[0].author().email, sides[1].author().email);
    assert_eq!(sides[0].author().timestamp, sides[1].author().timestamp);
    assert_ne!(sides[0].author().name, sides[1].author().name);
    let solved = converge_authors(repo, sides, None)?;
    assert_eq!(
        solved,
        ConvergedAttribute::Unsolved {
            base_commit: base.id().clone(),
            excluded_divergent_commits: HashSet::default(),
        }
    );

    // Same name and timestamp, different emails.
    let test_repo = TestRepo::init();
    let email_b = fixed_signature("Alice", "bob@example.com", 1_000, 0);
    let email_c = fixed_signature("Alice", "carol@example.com", 1_000, 0);
    let (repo, base, sides) = fork_change_with_authors(&test_repo, &author_a, &[email_b, email_c])?;
    assert_eq!(sides[0].author().name, sides[1].author().name);
    assert_eq!(sides[0].author().timestamp, sides[1].author().timestamp);
    assert_ne!(sides[0].author().email, sides[1].author().email);
    let solved = converge_authors(repo, sides, None)?;
    assert_eq!(
        solved,
        ConvergedAttribute::Unsolved {
            base_commit: base.id().clone(),
            excluded_divergent_commits: HashSet::default(),
        }
    );
    Ok(())
}

#[test]
fn test_converge_author_explicit_override_is_unchanged() -> TestResult {
    let test_repo = TestRepo::init();
    let author_a = fixed_signature("Alice", "alice@example.com", 1_000, 0);
    let author_b = fixed_signature("Bob", "bob@example.com", 2_000, 0);
    let author_c = fixed_signature("Carol", "carol@example.com", 3_000, 60);
    let (repo, _base, sides) =
        fork_change_with_authors(&test_repo, &author_a, &[author_b, author_c])?;
    let override_author = fixed_signature("Override", "override@example.com", 9_001, -210);

    let solved = converge_authors(repo, sides, Some(override_author.clone()))?;
    assert_eq!(solved, ConvergedAttribute::Solved(override_author));
    Ok(())
}
