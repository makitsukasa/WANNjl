function select_connection_cood(nIn, nOut, nHid)
	nIn += 1 # bias
	# println(nIn * nHid + nIn * nOut + nHid * nOut + div(nHid * (nHid - 1), 2))
	r = rand(1:nIn * nHid + nIn * nOut + nHid * nOut + div(nHid * (nHid - 1), 2))
	r = 13
	println(r)
	if r <= nIn * (nHid + nOut)
		x = div(r - 1, nIn) + nIn + 1
		y = mod(r - 1, nIn) + 1
		# println("i,h or i,o ", x, y)
		return y, x
	end
	r -= nIn * (nHid + nOut)
	if r <= nHid * nOut
		x = div(r - 1, nHid) + nIn + nHid + 1
		y = mod(r - 1, nHid) + nIn + 1
		# println("h,o ", x, y)
		return y, x
	end
	r -= nHid * nOut
	w = nHid - 1
	for w = nHid-1:-1:1
		if r > w
			r -= w
			continue
		end
		x = nIn + nHid - r + 1
		y = nIn + nHid - w
		println("r: ", r, ", w: ", w)
		println("h,h ", x, ", ", y)
		return y, x
	end
end

# 1にしていい座標を1こ返す
function select_connection_index(nIn::Int, nHid::Int, nOut::Int)::CartesianIndex{2}
	nIn += 1 # bias
	# println(nIn * nHid + nIn * nOut + nHid * nOut + div(nHid(nHid - 1), 2))
	r = rand(1:nIn * nHid + nIn * nOut + nHid * nOut + div(nHid(nHid - 1), 2))
	if r <= nIn * (nHid + nOut)
		x = div(r - 1, nIn) + nIn + 1
		y = mod(r - 1, nIn) + 1
		# println(nIn, " ", nHid, " ", nOut, " ", r)
		# println("i,h or i,o ", x, y)
		return CartesianIndex(y, x)
	end
	r -= nIn * (nHid + nOut)
	if r <= nHid * nOut
		x = div(r - 1, nHid) + nIn + nHid + 1
		y = mod(r - 1, nHid) + nIn + 1
		# println(nIn, " ", nHid, " ", nOut, " ", r)
		# println("h,o ", x, y)
		return CartesianIndex(y, x)
	end
	r -= nHid * nOut
	w = nHid - 1
	for w = nHid-1:-1:1
		if r > w
			r -= w
			continue
		end
		x = nIn + nHid - r + 1
		y = nIn + nHid - w
		# println("r: ", r, ", w: ", w)
		# println("h,h ", x, ", ", y)
		return CartesianIndex(y, x)
	end
	throw(error("error in function select_connection_index()"))
end
