import torch


def calculate_distances(
    embeddings_i: torch.Tensor, embeddings_j: torch.Tensor
) -> torch.Tensor:
    """Calculate distances between embeddings.

    Parameters
    ----------
    embeddings_i : torch.Tensor
        Embeddings of shape [B, N_feat], where B is a
        batch size, and N_feat is a number of features in
        a vector.
    embeddings_j : torch.Tensor
        Embeddings of shape [B, N_feat], where B is a
        batch size, and N_feat is a number of features in
        a vector.

    Returns
    -------
    torch.Tensor
        Output of shape [B], with the value as a distance
        between each pair.
    """
    return torch.sqrt(torch.sum(torch.square(embeddings_i - embeddings_j)))


def true_accept(
    embeddings_i: torch.Tensor,
    embeddings_j: torch.Tensor,
    labels: torch.Tensor,
    distance: torch.Tensor,
) -> torch.Tensor:
    """Calculate the true acceptance.

    True Acceptance (TA for short) shows how many actually
    similar pairs model correctly identified.

    Parameters
    ----------
    embeddings_i : torch.Tensor
        Embeddings of shape [B, N_feat], where B is a
        batch size, and N_feat is a number of features in
        a vector.
    embeddings_j : torch.Tensor
        Embeddings of shape [B, N_feat], where B is a
        batch size, and N_feat is a number of features in
        a vector.
    labels : torch.Tensor
        Tensor of shape [B] with values 1 if corresponding
        pair is indeed similar, and value 0 if corresponding pair
        is not similar.
    distance : torch.Tensor
        Distance to consider while calculating similarity
        between the embeddings. Used in L2 distance decision.

    Returns
    -------
    torch.Tensor
        A single value that corresponds to the metric.
    """
    distances = calculate_distances(embeddings_i, embeddings_j)

    predictions = distances < distance

    predictions = predictions.type(dtype=torch.int32)

    return sum(predictions & labels)


def false_accept(
    embeddings_i: torch.Tensor,
    embeddings_j: torch.Tensor,
    labels: torch.Tensor,
    distance: torch.Tensor,
) -> torch.Tensor:
    """Calculate the false acceptance.

    False Acceptance (FA for short) shows how many pairs
    were classified incorrectly as same.

    Parameters
    ----------
    embeddings_i : torch.Tensor
        Embeddings of shape [B, N_feat], where B is a
        batch size, and N_feat is a number of features in
        a vector.
    embeddings_j : torch.Tensor
        Embeddings of shape [B, N_feat], where B is a
        batch size, and N_feat is a number of features in
        a vector.
    labels : torch.Tensor
        Tensor of shape [B] with values 1 if corresponding
        pair is indeed similar, and value 0 if corresponding pair
        is not similar.
    distance : torch.Tensor
        Distance to consider while calculating similarity
        between the embeddings. Used in L2 distance decision.

    Returns
    -------
    torch.Tensor
        A single value that corresponds to the metric.
    """
    distances = calculate_distances(embeddings_i, embeddings_j)

    predictions = distances < distance

    predictions = predictions.type(dtype=torch.int32)

    return sum((predictions == 1) & (labels == 0))


def validation_rate(
    embeddings_i: torch.Tensor,
    embeddings_j: torch.Tensor,
    labels: torch.Tensor,
    distance: torch.Tensor,
) -> torch.Tensor:
    """Calculate validation rate.

    Validation Rate (VAL for short) is a percentage of all
    similar pairs captured.

    Parameters
    ----------
    embeddings_i : torch.Tensor
        Embeddings of shape [B, N_feat], where B is a
        batch size, and N_feat is a number of features in
        a vector.
    embeddings_j : torch.Tensor
        Embeddings of shape [B, N_feat], where B is a
        batch size, and N_feat is a number of features in
        a vector.
    labels : torch.Tensor
        Tensor of shape [B] with values 1 if corresponding
        pair is indeed similar, and value 0 if corresponding pair
        is not similar.
    distance : torch.Tensor
        Distance to consider while calculating similarity
        between the embeddings. Used in L2 distance decision.

    Returns
    -------
    torch.Tensor
        A single value that corresponds to the metric.
    """
    true_acceptance = true_accept(embeddings_i, embeddings_j, labels, distance)

    return true_acceptance / sum(labels == 1)


def false_rate(
    embeddings_i: torch.Tensor,
    embeddings_j: torch.Tensor,
    labels: torch.Tensor,
    distance: torch.Tensor,
) -> torch.Tensor:
    """Calculate false rate.

    False Rate (FAR for short) is a percentage of all
    different pairs not captured.

    Parameters
    ----------
    embeddings_i : torch.Tensor
        Embeddings of shape [B, N_feat], where B is a
        batch size, and N_feat is a number of features in
        a vector.
    embeddings_j : torch.Tensor
        Embeddings of shape [B, N_feat], where B is a
        batch size, and N_feat is a number of features in
        a vector.
    labels : torch.Tensor
        Tensor of shape [B] with values 1 if corresponding
        pair is indeed similar, and value 0 if corresponding pair
        is not similar.
    distance : torch.Tensor
        Distance to consider while calculating similarity
        between the embeddings. Used in L2 distance decision.

    Returns
    -------
    torch.Tensor
        A single value that corresponds to the metric.
    """
    false_acceptance = false_accept(embeddings_i, embeddings_j, labels, distance)

    return false_acceptance / sum(labels == 0)
